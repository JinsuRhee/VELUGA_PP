// save.cxx — HDF5 catalog writer, mirrors rv_save.pro / save_cat_conly.c
//
// All per-galaxy data fields are always written even when the corresponding
// computation was skipped — placeholder value is -1.0 / -1 in that case.

#include "pipeline/save.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef VPP_USE_HDF5
#include "hdf5.h"
#endif

namespace Pipeline {

#ifdef VPP_USE_HDF5

// ---------------------------------------------------------------------------
// HDF5 write helpers  (mirrors save_cat_write_d / _d2 / _i / _i64 / _s)
// ---------------------------------------------------------------------------
static void write_d(hid_t loc, const char *name, const double *data, hsize_t n)
{
    hid_t sp = H5Screate_simple(1, &n, nullptr);
    hid_t ds = H5Dcreate(loc, name, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds); H5Sclose(sp);
}

static void write_d2(hid_t loc, const char *name, const double *data,
                     hsize_t rows, hsize_t cols)
{
    hsize_t dims[2] = {rows, cols};
    hid_t sp = H5Screate_simple(2, dims, nullptr);
    hid_t ds = H5Dcreate(loc, name, H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds); H5Sclose(sp);
}

static void write_i32(hid_t loc, const char *name, const int32_t *data, hsize_t n)
{
    hid_t sp = H5Screate_simple(1, &n, nullptr);
    hid_t ds = H5Dcreate(loc, name, H5T_NATIVE_INT32, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds); H5Sclose(sp);
}

static void write_i64(hid_t loc, const char *name, const int64_t *data, hsize_t n)
{
    hid_t sp = H5Screate_simple(1, &n, nullptr);
    hid_t ds = H5Dcreate(loc, name, H5T_NATIVE_INT64, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds); H5Sclose(sp);
}

// Write a string array as a fixed-length ASCII dataset (like Flux_List)
static void write_strlist(hid_t loc, const char *name,
                          const std::vector<std::string> &strs)
{
    // Always write at least one entry so the dataset exists
    const std::vector<std::string> &src = strs.empty()
        ? std::vector<std::string>{""} : strs;

    size_t maxlen = 0;
    for (auto &s : src) maxlen = std::max(maxlen, s.size());
    maxlen += 1; // null terminator

    hid_t st = H5Tcopy(H5T_C_S1);
    H5Tset_size(st, maxlen);
    H5Tset_strpad(st, H5T_STR_NULLTERM);
    H5Tset_cset(st, H5T_CSET_ASCII);

    hsize_t n = src.size();
    std::vector<char> flat(n * maxlen, '\0');
    for (hsize_t i = 0; i < n; ++i)
        std::strncpy(&flat[i * maxlen], src[i].c_str(), maxlen);

    hid_t sp = H5Screate_simple(1, &n, nullptr);
    hid_t ds = H5Dcreate(loc, name, st, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, st, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Dclose(ds); H5Sclose(sp); H5Tclose(st);
}

// Convenience: write scalar double
static inline void write_d1(hid_t loc, const char *name, double v)
{
    write_d(loc, name, &v, 1);
}
// Convenience: write scalar int32
static inline void write_i1(hid_t loc, const char *name, int32_t v)
{
    write_i32(loc, name, &v, 1);
}

#endif // VPP_USE_HDF5

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
void save_result(
    const vpp_set::Settings &vh,
    int32_t                  snap,
    const RawCatalog        &cat,
    const PtclIDCatalog     &ptcl,
    const BPropResult       &bprop,
    const GasPropResult     &gasprop,
    const CosmoInfo         &cosmo)
{
#ifndef VPP_USE_HDF5
    LOG() << "[save] HDF5 not compiled in — skipping";
    (void)snap; (void)cat; (void)ptcl; (void)bprop; (void)gasprop; (void)cosmo;
    return;
#else
    // Build output path: dir_catalog/{Galaxy/VR_Galaxy|Halo/VR_Halo}/snap_NNNN.hdf5
    const std::string horg_dir = (vh.horg == 'g') ? "Galaxy/VR_Galaxy" : "Halo/VR_Halo";
    char snapbuf[32];
    std::snprintf(snapbuf, sizeof(snapbuf), "snap_%04d.hdf5", snap);
    const std::string outdir = vh.dir_catalog + "/" + horg_dir;
    const std::string fname  = outdir + "/" + snapbuf;

    // Ensure directory exists (mkdir -p equivalent for one level deep)
    ::mkdir(outdir.c_str(), 0755);

    LOG() << "[save] writing " << fname;

    hid_t fid = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        LOG() << "[save] ERROR: H5Fcreate failed for " << fname;
        return;
    }

    const int64_t n_gal    = cat.n;
    const int32_t n_sfr    = bprop.n_sfr;
    const int32_t n_mag_ap = bprop.n_mag_ap;
    const int32_t n_flux   = bprop.n_flux;
    const int32_t n_conf_ap = bprop.n_conf_ap;
    const int32_t n_gas_ap = gasprop.n_gas_ap;
    const bool has_gasprop  = (gasprop.n_gal == n_gal && n_gas_ap > 0
                                && !gasprop.gm_tot_tot.empty());

    static constexpr double NULLD   = -1.0;
    static constexpr int32_t NULLI  = -1;

    // =========================================================================
    // Top-level header datasets
    // =========================================================================

    // SFR aperture info
    if (!vh.sfr_r.empty()) {
        write_d(fid, "SFR_R", vh.sfr_r.data(), vh.sfr_r.size());
        write_d(fid, "SFR_T", vh.sfr_t.data(), vh.sfr_t.size());
    } else {
        write_d1(fid, "SFR_R", NULLD);
        write_d1(fid, "SFR_T", NULLD);
    }

    // Photometry aperture info
    if (!vh.mag_r.empty())
        write_d(fid, "MAG_R", vh.mag_r.data(), vh.mag_r.size());
    else
        write_d1(fid, "MAG_R", NULLD);

    // Contamination fraction aperture info
    if (!vh.conf_r.empty())
        write_d(fid, "CONF_R", vh.conf_r.data(), vh.conf_r.size());
    else
        write_d1(fid, "CONF_R", NULLD);

    // Band names
    write_strlist(fid, "Flux_List", vh.flux_list);

    // Gas aperture info (new fields)
    if (!vh.gas_r.empty())
        write_d(fid, "GAS_R", vh.gas_r.data(), vh.gas_r.size());
    else
        write_d1(fid, "GAS_R", NULLD);
    write_d1(fid, "GRAV_R", vh.grav_r);

    // Bulk catalog arrays [n_gal]
    auto get_f64 = [&](const char *col) -> std::vector<double> {
        auto it = cat.f64.find(col);
        if (it != cat.f64.end()) return it->second;
        return std::vector<double>(n_gal, NULLD);
    };

    auto mass_tot = get_f64("mass_tot");
    auto r_half   = get_f64("R_HalfMass");
    auto mvir     = get_f64("Mvir");
    auto rvir     = get_f64("Rvir");
    auto m200     = get_f64("mass_200crit");
    auto r200     = get_f64("R_200crit");

    write_d(fid, "Mass_tot",     mass_tot.data(), (hsize_t)n_gal);
    write_d(fid, "R_HalfMass",   r_half.data(),   (hsize_t)n_gal);
    write_d(fid, "Mvir",         mvir.data(),     (hsize_t)n_gal);
    write_d(fid, "Rvir",         rvir.data(),     (hsize_t)n_gal);
    write_d(fid, "Mass_200crit", m200.data(),     (hsize_t)n_gal);
    write_d(fid, "R_200crit",    r200.data(),     (hsize_t)n_gal);

    // ID as int32 (matching save_cat_conly.c which uses H5T_NATIVE_INT)
    {
        std::vector<int32_t> ids(n_gal, NULLI);
        if (cat.i64.count("ID"))
            for (int64_t i = 0; i < n_gal; ++i)
                ids[i] = (int32_t)cat.i64.at("ID")[i];
        else if (cat.i32.count("ID"))
            ids = cat.i32.at("ID");
        write_i32(fid, "ID", ids.data(), (hsize_t)n_gal);
    }

    // CONF_M / CONF_N: 2-D [n_gal × n_conf_ap] matching save_cat_write_d2
    if (n_conf_ap > 0 && (int64_t)bprop.confrac_m.size() == n_gal * n_conf_ap) {
        write_d2(fid, "CONF_M", bprop.confrac_m.data(), (hsize_t)n_gal, (hsize_t)n_conf_ap);
        write_d2(fid, "CONF_N", bprop.confrac_n.data(), (hsize_t)n_gal, (hsize_t)n_conf_ap);
    } else {
        write_d1(fid, "CONF_M", NULLD);
        write_d1(fid, "CONF_N", NULLD);
    }

    // =========================================================================
    // Per-galaxy groups
    // =========================================================================

    for (int64_t gi = 0; gi < n_gal; ++gi) {

        // Galaxy ID for path
        int32_t gid = NULLI;
        if (cat.i64.count("ID")) gid = (int32_t)cat.i64.at("ID")[gi];
        else if (cat.i32.count("ID")) gid = cat.i32.at("ID")[gi];

        char idbase[64], gpath[96], ppath[96];
        std::snprintf(idbase, sizeof(idbase), "/ID_%06d", gid);
        std::snprintf(gpath,  sizeof(gpath),  "/ID_%06d/G_Prop", gid);
        std::snprintf(ppath,  sizeof(ppath),  "/ID_%06d/P_Prop", gid);

        hid_t grp   = H5Gcreate(fid, idbase, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t grp_g = H5Gcreate(fid, gpath,  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t grp_p = H5Gcreate(fid, ppath,  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // ----- /ID_XXXXXX/ scalars -----
        write_d1(grp, "Aexp", cosmo.aexp);

        {
            int32_t ic = NULLI;
            if (!bprop.isclump.empty()) ic = bprop.isclump[(size_t)gi];
            write_i1(grp, "isclump", ic);
        }

        // ----- P_Prop: particle IDs -----
        {
            bool have_ptcl = (ptcl.b_ind.size() >= (size_t)(gi * 2 + 2));
            if (have_ptcl) {
                int64_t bs = ptcl.b_ind[(size_t)(gi * 2 + 0)];
                int64_t be = ptcl.b_ind[(size_t)(gi * 2 + 1)];
                int64_t us = ptcl.u_ind[(size_t)(gi * 2 + 0)];
                int64_t ue = ptcl.u_ind[(size_t)(gi * 2 + 1)];
                int64_t nb = (be >= bs) ? be - bs + 1 : 0;
                int64_t nu = (ue >= us) ? ue - us + 1 : 0;

                std::vector<int64_t> ids;
                ids.reserve((size_t)(nb + nu));
                for (int64_t k = 0; k < nb; ++k) ids.push_back(ptcl.p_id[(size_t)(bs + k)]);
                for (int64_t k = 0; k < nu; ++k) ids.push_back(ptcl.p_id[(size_t)(us + k)]);

                // Filter sentinel values (same threshold as save_cat_conly.c)
                std::vector<int64_t> filt;
                filt.reserve(ids.size());
                for (auto id : ids)
                    if (id > -922337203685477580LL) filt.push_back(id);

                if (!filt.empty())
                    write_i64(grp_p, "P_ID", filt.data(), (hsize_t)filt.size());
            }
        }

        // ----- G_Prop: raw catalog columns (one scalar per galaxy) -----
        for (auto &[col, vals] : cat.i64) {
            char dname[96]; std::snprintf(dname, sizeof(dname), "G_%s", col.c_str());
            int64_t v = vals[(size_t)gi];
            write_i64(grp_g, dname, &v, 1);
        }
        for (auto &[col, vals] : cat.i32) {
            char dname[96]; std::snprintf(dname, sizeof(dname), "G_%s", col.c_str());
            int32_t v = vals[(size_t)gi];
            write_i32(grp_g, dname, &v, 1);
        }
        for (auto &[col, vals] : cat.f64) {
            char dname[96]; std::snprintf(dname, sizeof(dname), "G_%s", col.c_str());
            double v = vals[(size_t)gi];
            write_d(grp_g, dname, &v, 1);
        }

        // ----- G_Prop: SFR [n_sfr] -----
        if (n_sfr > 0 && (int64_t)bprop.sfr.size() == n_gal * n_sfr)
            write_d(grp_g, "G_SFR", &bprop.sfr[(size_t)(gi * n_sfr)], (hsize_t)n_sfr);
        else
            write_d1(grp_g, "G_SFR", NULLD);

        // ----- G_Prop: ABmag + SB per flux band -----
        // abmag/sbf layout: [gal * n_mag_ap * n_flux + ap * n_flux + fi]
        for (int32_t fi = 0; fi < n_flux; ++fi) {
            const std::string &band = vh.flux_list[(size_t)fi];
            char dname[96];

            std::snprintf(dname, sizeof(dname), "G_ABmag_%s", band.c_str());
            if (n_mag_ap > 0 && (int64_t)bprop.abmag.size() == n_gal * n_mag_ap * n_flux) {
                std::vector<double> tmp((size_t)n_mag_ap);
                for (int32_t ai = 0; ai < n_mag_ap; ++ai)
                    tmp[(size_t)ai] = bprop.abmag[(size_t)(gi * n_mag_ap * n_flux
                                                            + ai * n_flux + fi)];
                write_d(grp_g, dname, tmp.data(), (hsize_t)n_mag_ap);
            } else {
                write_d1(grp_g, dname, NULLD);
            }

            std::snprintf(dname, sizeof(dname), "G_SB_%s", band.c_str());
            if (n_mag_ap > 0 && (int64_t)bprop.sbf.size() == n_gal * n_mag_ap * n_flux) {
                std::vector<double> tmp((size_t)n_mag_ap);
                for (int32_t ai = 0; ai < n_mag_ap; ++ai)
                    tmp[(size_t)ai] = bprop.sbf[(size_t)(gi * n_mag_ap * n_flux
                                                          + ai * n_flux + fi)];
                write_d(grp_g, dname, tmp.data(), (hsize_t)n_mag_ap);
            } else {
                write_d1(grp_g, dname, NULLD);
            }
        }

        // ----- G_Prop: contamination fractions -----
        if (n_conf_ap > 0 && (int64_t)bprop.confrac_m.size() == n_gal * n_conf_ap) {
            write_d(grp_g, "G_ConFrac_M",
                    &bprop.confrac_m[(size_t)(gi * n_conf_ap)], (hsize_t)n_conf_ap);
            write_d(grp_g, "G_ConFrac_N",
                    &bprop.confrac_n[(size_t)(gi * n_conf_ap)], (hsize_t)n_conf_ap);
        } else {
            write_d1(grp_g, "G_ConFrac_M", NULLD);
            write_d1(grp_g, "G_ConFrac_N", NULLD);
        }

        // ----- G_Prop: gas masses [n_gas_ap] and pressures -----
        if (has_gasprop) {
            const size_t base = (size_t)(gi * n_gas_ap);
            write_d(grp_g, "G_GasMass_tot_tot",    &gasprop.gm_tot_tot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_tot_hot",    &gasprop.gm_tot_hot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_tot_cold",   &gasprop.gm_tot_cold[base], (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_bound_tot",  &gasprop.gm_ism_tot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_bound_hot",  &gasprop.gm_ism_hot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_bound_cold", &gasprop.gm_ism_cold[base], (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_cgm_tot",    &gasprop.gm_cgm_tot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_cgm_hot",    &gasprop.gm_cgm_hot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_cgm_cold",   &gasprop.gm_cgm_cold[base], (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_icm_tot",    &gasprop.gm_icm_tot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_icm_hot",    &gasprop.gm_icm_hot[base],  (hsize_t)n_gas_ap);
            write_d(grp_g, "G_GasMass_icm_cold",   &gasprop.gm_icm_cold[base], (hsize_t)n_gas_ap);
            write_d1(grp_g, "G_RamPressure",  gasprop.ram_pressure[(size_t)gi]);
            write_d1(grp_g, "G_GravPressure", gasprop.grav_pressure[(size_t)gi]);
        } else {
            write_d1(grp_g, "G_GasMass_tot_tot",    NULLD);
            write_d1(grp_g, "G_GasMass_tot_hot",    NULLD);
            write_d1(grp_g, "G_GasMass_tot_cold",   NULLD);
            write_d1(grp_g, "G_GasMass_bound_tot",  NULLD);
            write_d1(grp_g, "G_GasMass_bound_hot",  NULLD);
            write_d1(grp_g, "G_GasMass_bound_cold", NULLD);
            write_d1(grp_g, "G_GasMass_cgm_tot",    NULLD);
            write_d1(grp_g, "G_GasMass_cgm_hot",    NULLD);
            write_d1(grp_g, "G_GasMass_cgm_cold",   NULLD);
            write_d1(grp_g, "G_GasMass_icm_tot",    NULLD);
            write_d1(grp_g, "G_GasMass_icm_hot",    NULLD);
            write_d1(grp_g, "G_GasMass_icm_cold",   NULLD);
            write_d1(grp_g, "G_RamPressure",  NULLD);
            write_d1(grp_g, "G_GravPressure", NULLD);
        }

        H5Gclose(grp_g);
        H5Gclose(grp_p);
        H5Gclose(grp);
    }

    H5Fclose(fid);
    LOG() << "[save] done";

#endif // VPP_USE_HDF5
}

} // namespace Pipeline
