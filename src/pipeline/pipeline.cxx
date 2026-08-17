#include "pipeline/pipeline.h"
#include "pipeline/bprop.h"
#include "pipeline/cosmo.h"
#include "pipeline/gasprop.h"
#include "pipeline/miles.h"
#include "pipeline/save.h"

#include <algorithm>
#include <array>
#include <dirent.h>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace
{

enum class ColType { I64, I32, F64 };

ColType get_col_type(const std::string &name)
{
    static const std::set<std::string> i64 = {"ID", "ID_mbp", "ID_minpot", "hostHaloID"};
    static const std::set<std::string> i32 = {"numSubStruct", "npart", "Structuretype",
                                               "n_gas", "n_star", "n_bh"};
    if (i64.count(name)) return ColType::I64;
    if (i32.count(name)) return ColType::I32;
    return ColType::F64;
}

std::string strip_annotation(const std::string &tok)
{
    auto p = tok.find('(');
    return (p != std::string::npos) ? tok.substr(0, p) : tok;
}

std::vector<std::string> split_ws(const std::string &s)
{
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

int64_t count_lines(const std::string &path)
{
    std::ifstream in(path);
    int64_t n = 0;
    std::string line;
    while (std::getline(in, line)) ++n;
    return n;
}

// Collect files matching base_prefix + ".dat." + mid + ".*", sorted numerically by suffix
std::vector<std::string> collect_vr_files(const std::string &sdir,
                                           const std::string &base,
                                           const std::string &mid)
{
    const std::string needle = base + ".dat." + mid + ".";
    std::vector<std::pair<int32_t, std::string>> numbered;

    DIR *d = ::opendir(sdir.c_str());
    if (!d) return {};
    struct dirent *ent;
    while ((ent = ::readdir(d)) != nullptr) {
        std::string fn = ent->d_name;
        if (fn.find(needle) != 0) continue;
        std::string suffix = fn.substr(needle.size());
        // exclude unbound files when looking for bound: "catalog_particles.N" but not
        // "catalog_particles.unbound.N"
        if (mid == "catalog_particles" && fn.find("unbound") != std::string::npos) continue;
        try {
            int32_t idx = std::stoi(suffix);
            numbered.push_back({idx, sdir + "/" + fn});
        } catch (...) {}
    }
    ::closedir(d);
    std::sort(numbered.begin(), numbered.end());
    std::vector<std::string> out;
    for (auto &p : numbered) out.push_back(p.second);
    return out;
}

} // namespace

namespace Pipeline
{

//=============================================================================
// load_catalog — rv_RawCatalog
//=============================================================================
RawCatalog load_catalog(const vpp_set::Settings &vh, int32_t snap)
{
    const std::string sdir     = snap_dir(vh, snap);
    const std::string base     = vh.horg == 'g' ? "galaxy" : "halo";
    const std::string hdr_file = sdir + "/" + base + ".dat.properties.0";

    LOG() << "[load_catalog] snap=" << snap;

    // 1. Column names from 3rd header line
    std::vector<std::string> col_names;
    {
        std::ifstream in(hdr_file);
        if (!in) { LOG() << "  ERROR: cannot open " << hdr_file; return {}; }
        std::string line;
        std::getline(in, line);
        std::getline(in, line);
        std::getline(in, line);
        for (auto &tok : split_ws(line))
            col_names.push_back(strip_annotation(tok));
    }
    const int32_t n_col = (int32_t)col_names.size();

    // 2. Which columns to keep
    std::vector<bool> keep(n_col, vh.column_list.empty());
    if (!vh.column_list.empty())
        for (int32_t c = 0; c < n_col; ++c)
            for (auto &req : vh.column_list)
                if (col_names[c] == req) { keep[c] = true; break; }

    // 3. Find all *.dat.properties.* files
    auto fnames = collect_vr_files(sdir, base, "properties");
    LOG() << "  files=" << fnames.size() << "  columns=" << n_col;

    // 4. Count total rows
    int64_t n_total = 0;
    for (auto &fn : fnames) { int64_t nl = count_lines(fn); if (nl > 3) n_total += nl - 3; }
    LOG() << "  total rows=" << n_total;

    // 5. Allocate
    RawCatalog cat;
    cat.n = n_total;
    for (int32_t c = 0; c < n_col; ++c) {
        if (!keep[c]) continue;
        switch (get_col_type(col_names[c])) {
            case ColType::I64: cat.i64[col_names[c]].resize(n_total, 0);   break;
            case ColType::I32: cat.i32[col_names[c]].resize(n_total, 0);   break;
            case ColType::F64: cat.f64[col_names[c]].resize(n_total, 0.0); break;
        }
    }

    // 6. Read
    int64_t row = 0;
    for (auto &fn : fnames) {
        std::ifstream in(fn);
        std::string line;
        std::getline(in, line); std::getline(in, line); std::getline(in, line);
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            auto toks = split_ws(line);
            if ((int32_t)toks.size() < n_col) continue;
            for (int32_t c = 0; c < n_col; ++c) {
                if (!keep[c]) continue;
                switch (get_col_type(col_names[c])) {
                    case ColType::I64: cat.i64[col_names[c]][row] = std::stoll(toks[c]); break;
                    case ColType::I32: cat.i32[col_names[c]][row] = (int32_t)std::stol(toks[c]); break;
                    case ColType::F64: cat.f64[col_names[c]][row] = std::stod(toks[c]); break;
                }
            }
            ++row;
        }
    }

    LOG() << "  loaded n=" << cat.n;
    return cat;
}

//=============================================================================
// read_ptclid — rv_ReadID
//
// File formats
//   catalog_groups.N      line1: "# {n_mpi}"
//                         line2: "{n_obj_this_file} {n_tot_global}"
//                         then three sections of n_obj lines each:
//                           section A: npart_total per object
//                           section B: bound start index per object (0-based from data start)
//                           section C: unbound start index per object
//
//   catalog_particles.N   line1: header
//                         line2: "{local_count} {global_total}"
//                         lines 3+: particle IDs, one per line, in object order
//
//   catalog_particles.unbound.N   same format as catalog_particles
//=============================================================================
PtclIDCatalog read_ptclid(const vpp_set::Settings &vh, int32_t snap)
{
    const std::string sdir = snap_dir(vh, snap);
    const std::string base = vh.horg == 'g' ? "galaxy" : "halo";

    LOG() << "[read_ptclid] snap=" << snap;

    auto grp_files = collect_vr_files(sdir, base, "catalog_groups");
    auto bdn_files = collect_vr_files(sdir, base, "catalog_particles");
    // unbound: must match "catalog_particles.unbound.*"
    std::vector<std::string> ubd_files;
    {
        const std::string needle = base + ".dat.catalog_particles.unbound.";
        std::vector<std::pair<int32_t, std::string>> numbered;
        if (DIR *d2 = ::opendir(sdir.c_str())) {
            struct dirent *e2;
            while ((e2 = ::readdir(d2)) != nullptr) {
                std::string fn = e2->d_name;
                if (fn.find(needle) != 0) continue;
                std::string sfx = fn.substr(needle.size());
                try { numbered.push_back({std::stoi(sfx), sdir + "/" + fn}); } catch (...) {}
            }
            ::closedir(d2);
        }
        std::sort(numbered.begin(), numbered.end());
        for (auto &p : numbered) ubd_files.push_back(p.second);
    }

    const int32_t n_mpi = (int32_t)grp_files.size();
    LOG() << "  MPI files=" << n_mpi;

    if (n_mpi == 0) return {};

    //-----
    // Phase 1: read all catalog_groups files → build n_obj_per_file and n_part table
    // n_part[global_obj_idx] = {file_idx, npart_all, bdn_start, ubd_start}
    //-----
    int32_t n_tot = 0;
    std::vector<int32_t> n_obj_per_file(n_mpi, 0);

    // first pass: get n_tot from first file's line 2
    {
        std::ifstream in(grp_files[0]);
        std::string line;
        std::getline(in, line);                 // line 1: "# n_mpi"
        std::getline(in, line);                 // line 2: "n_obj n_tot"
        auto toks = split_ws(line);
        n_obj_per_file[0] = std::stoi(toks[0]);
        n_tot             = std::stoi(toks[1]);
    }
    for (int32_t i = 1; i < n_mpi; ++i) {
        std::ifstream in(grp_files[i]);
        std::string line;
        std::getline(in, line);
        std::getline(in, line);
        n_obj_per_file[i] = std::stoi(split_ws(line)[0]);
    }

    // n_part: [n_tot][4]  cols: 0=file, 1=npart_all, 2=bdn_start, 3=ubd_start
    std::vector<std::array<int64_t, 4>> n_part(n_tot);

    {
        int32_t i0 = 0;
        for (int32_t i = 0; i < n_mpi; ++i) {
            const int32_t nobj = n_obj_per_file[i];
            if (nobj == 0) continue;

            std::ifstream in(grp_files[i]);
            std::string line;
            std::getline(in, line);   // skip line 1
            std::getline(in, line);   // skip line 2

            // mark file index
            for (int32_t j = 0; j < nobj; ++j) n_part[i0 + j][0] = i;

            // section A: npart_all
            for (int32_t j = 0; j < nobj; ++j) {
                std::getline(in, line);
                n_part[i0 + j][1] = std::stoll(line);
            }
            // section B: bdn_start
            for (int32_t j = 0; j < nobj; ++j) {
                std::getline(in, line);
                n_part[i0 + j][2] = std::stoll(line);
            }
            // section C: ubd_start
            for (int32_t j = 0; j < nobj; ++j) {
                std::getline(in, line);
                n_part[i0 + j][3] = std::stoll(line);
            }
            i0 += nobj;
        }
    }

    //-----
    // Phase 2: read bound particle IDs
    //-----
    PtclIDCatalog cat;
    cat.n_obj = n_tot;
    cat.b_ind.resize((size_t)n_tot * 2, -1);
    cat.u_ind.resize((size_t)n_tot * 2, -1);

    {
        int32_t i0 = 0;
        int64_t j0 = 0;   // global bound counter
        for (int32_t i = 0; i < n_mpi; ++i) {
            const int32_t nobj = n_obj_per_file[i];
            if (nobj == 0) { i0 += nobj; continue; }

            const int64_t bdn_nlines = count_lines(bdn_files[i]);

            std::ifstream bdn(bdn_files[i]);
            std::string line;
            std::getline(bdn, line);   // skip header 1
            std::getline(bdn, line);   // skip header 2

            for (int32_t j = 0; j < nobj; ++j) {
                const int64_t gj = i0 + j;
                int64_t n_bnd;
                if (j < nobj - 1)
                    n_bnd = n_part[gj + 1][2] - n_part[gj][2];
                else
                    n_bnd = bdn_nlines - 2 - n_part[gj][2];

                cat.b_ind[gj * 2 + 0] = j0;
                for (int64_t k = 0; k < n_bnd; ++k) {
                    std::getline(bdn, line);
                    cat.p_id.push_back(std::stoll(line));
                }
                cat.b_ind[gj * 2 + 1] = j0 + n_bnd - 1;   // start > end when n_bnd==0
                j0 += n_bnd;
            }
            i0 += nobj;
        }
        cat.n_bdn = j0;
    }

    //-----
    // Phase 3: read unbound particle IDs (appended after bound in p_id)
    //-----
    {
        int32_t i0 = 0;
        int64_t k0 = 0;   // global unbound counter
        for (int32_t i = 0; i < n_mpi; ++i) {
            const int32_t nobj = n_obj_per_file[i];
            if (nobj == 0) { i0 += nobj; continue; }

            const int64_t ubd_nlines = count_lines(ubd_files[i]);

            std::ifstream ubd(ubd_files[i]);
            std::string line;
            std::getline(ubd, line);
            std::getline(ubd, line);

            for (int32_t j = 0; j < nobj; ++j) {
                const int64_t gj = i0 + j;
                int64_t n_ubd;
                if (j < nobj - 1)
                    n_ubd = n_part[gj + 1][3] - n_part[gj][3];
                else
                    n_ubd = ubd_nlines - 2 - n_part[gj][3];

                // u_ind offsets into the full p_id (after bound section)
                cat.u_ind[gj * 2 + 0] = cat.n_bdn + k0;
                for (int64_t k = 0; k < n_ubd; ++k) {
                    std::getline(ubd, line);
                    cat.p_id.push_back(std::stoll(line));
                }
                cat.u_ind[gj * 2 + 1] = cat.n_bdn + k0 + n_ubd - 1;
                k0 += n_ubd;
            }
            i0 += nobj;
        }
        cat.n_ubd = k0;
    }

    LOG() << "  n_obj=" << cat.n_obj
          << "  n_bdn=" << cat.n_bdn
          << "  n_ubd=" << cat.n_ubd;
    return cat;
}

//=============================================================================
// Stubs
//=============================================================================
void member_match(const vpp_set::Settings &vh, int32_t snap)
{
    LOG() << "[member_match] snap=" << snap << " (stub)";
    (void)vh;
}

void compute_bulk(const vpp_set::Settings &vh, int32_t snap,
                  const RawCatalog &cat, const PtclIDCatalog &ptcl)
{
    LOG() << "[compute_bulk] snap=" << snap;
    // cosmo always needed: confrac + gasprop use omega_m/omega_b for both horg modes
    CosmoInfo cosmo = (vh.newramses != 0)
                    ? read_info_hdf(vh.dir_raw, snap)
                    : read_info    (vh.dir_raw, snap);
    CosmTable ctbl;
    MilesML   miles;
    if (vh.horg == 'g' && !vh.skip_bprop) {
        ctbl  = build_cosm_table(cosmo.omega_m, cosmo.omega_l, cosmo.H0);
        miles = load_miles();
    }

    BPropResult bprop_res;
    if (!vh.skip_bprop || !vh.skip_confrac) {
        bprop_res = compute_bprop(vh, snap, cat, ptcl, cosmo, miles, ctbl);
        LOG() << "  bprop done: n_gal=" << bprop_res.n_gal
              << "  n_flux=" << bprop_res.n_flux
              << "  n_mag_ap=" << bprop_res.n_mag_ap
              << "  n_sfr=" << bprop_res.n_sfr
              << "  n_conf_ap=" << bprop_res.n_conf_ap;
    } else {
        LOG() << "  bprop + confrac skipped (skip_bprop=" << vh.skip_bprop
              << " skip_confrac=" << vh.skip_confrac << ")";
    }

    GasPropResult gasprop_res;
    if (!vh.skip_gasprop && !vh.gas_r.empty()) {
        gasprop_res = compute_gasprop(vh, snap, cat, cosmo);
        LOG() << "  gasprop done: n_gal=" << gasprop_res.n_gal
              << "  n_gas_ap=" << gasprop_res.n_gas_ap;
    } else if (vh.skip_gasprop) {
        LOG() << "  gasprop skipped";
    } else {
        LOG() << "  gasprop skipped (gas_r empty)";
    }

    save_result(vh, snap, cat, ptcl, bprop_res, gasprop_res, cosmo);
}

void save(const vpp_set::Settings &vh, int32_t snap)
{
    // save_result is called at the end of compute_bulk
    (void)vh; (void)snap;
}

} // namespace Pipeline
