// rawdata.cxx — RAMSES raw data reader (particles + AMR leaf cells)
//
// Old format (newramses==0): Fortran unformatted binary per-CPU
//   part_{snap:05d}.out{cpu:05d}   particles
//   amr_{snap:05d}.out{cpu:05d}    AMR octree
//   hydro_{snap:05d}.out{cpu:05d}  hydro variables
//
// New format (newramses==1): HDF5
//   {output_dir}/part_{snap:05d}.h5          particles (groups "star", "dm")
//   {dir_raw}/../hdf/cell_{snap:05d}.h5      leaf cells (group "leaf")

#include "pipeline/rawdata.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef VPP_USE_OMP
#  include <omp.h>
#endif

#ifdef VPP_USE_HDF5
#  include <hdf5.h>
#endif

namespace
{

// ============================================================
// Path helpers
// ============================================================

static std::string snap_dir(const vpp_set::Settings &vh, int32_t snap)
{
    std::ostringstream o;
    o << vh.dir_raw << "/output_" << std::setw(5) << std::setfill('0') << snap;
    return o.str();
}

static std::string fname_part(const std::string &sd, int32_t snap, int32_t cpu)
{
    std::ostringstream o;
    o << sd << "/part_" << std::setw(5) << std::setfill('0') << snap
             << ".out"   << std::setw(5) << std::setfill('0') << cpu;
    return o.str();
}

static std::string fname_amr(const std::string &sd, int32_t snap, int32_t cpu)
{
    std::ostringstream o;
    o << sd << "/amr_" << std::setw(5) << std::setfill('0') << snap
             << ".out"  << std::setw(5) << std::setfill('0') << cpu;
    return o.str();
}

static std::string fname_hydro(const std::string &sd, int32_t snap, int32_t cpu)
{
    std::ostringstream o;
    o << sd << "/hydro_" << std::setw(5) << std::setfill('0') << snap
              << ".out"    << std::setw(5) << std::setfill('0') << cpu;
    return o.str();
}

static int32_t count_cpu_files(const std::string &sd, int32_t snap)
{
    std::ostringstream o;
    o << "part_" << std::setw(5) << std::setfill('0') << snap << ".out";
    const std::string prefix = o.str();
    int32_t cnt = 0;
    DIR *d = ::opendir(sd.c_str());
    if (!d) return 0;
    struct dirent *ent;
    while ((ent = ::readdir(d)) != nullptr) {
        std::string fn = ent->d_name;
        if (fn.compare(0, prefix.size(), prefix) == 0) ++cnt;
    }
    ::closedir(d);
    return cnt;
}

// ============================================================
// Fortran unformatted binary record helpers
// ============================================================

static void frec_skip(FILE *fp)
{
    int32_t len;
    if (fread(&len, 4, 1, fp) != 1) return;
    fseek(fp, (long)len, SEEK_CUR);
    fread(&len, 4, 1, fp);
}

template <typename T>
static bool frec_read(FILE *fp, T *buf, int32_t n)
{
    int32_t len;
    if (fread(&len, 4, 1, fp) != 1) return false;
    if (fread(buf, sizeof(T), (size_t)n, fp) != (size_t)n) return false;
    fread(&len, 4, 1, fp);
    return true;
}

static int32_t frec_peek_len(FILE *fp)
{
    int32_t len = -1;
    if (fread(&len, 4, 1, fp) != 1) return -1;
    fseek(fp, -4, SEEK_CUR);
    return len;
}

// ============================================================
// Thread-local accumulator helpers
// ============================================================

struct StarLocal {
    std::vector<double>  x, y, z;
    std::vector<float>   vx, vy, vz, mass, age, metal;
    std::vector<int64_t> id;
};

struct DMLocal {
    std::vector<double>  x, y, z;
    std::vector<float>   vx, vy, vz, mass;
    std::vector<int64_t> id;
};

// nvar variables per cell, var[ivar][icell] layout
struct CellLocal {
    std::vector<double>  x, y, z;
    std::vector<float>   dx;
    std::vector<int32_t> level;
    int32_t              nvar = 0;
    std::vector<std::vector<double>> var;   // [nvar][ncell_local]

    void init(int32_t nv) {
        nvar = nv;
        var.assign((size_t)nv, std::vector<double>{});
    }
    int64_t n() const { return (int64_t)x.size(); }
    void push(double cx, double cy, double cz, float cdx, int32_t lev,
              const std::vector<double> &hd) {
        x.push_back(cx); y.push_back(cy); z.push_back(cz);
        dx.push_back(cdx); level.push_back(lev);
        for (int32_t v = 0; v < nvar; ++v)
            var[(size_t)v].push_back(hd[(size_t)v]);
    }
};

template <typename Vec, typename Src>
static void vec_append(Vec &dst, Src &src)
{ dst.insert(dst.end(), src.begin(), src.end()); }


// Merge thread-local CellLocal into the global column-major CellData.
// var is column-major flat [nvar * n], so appending a new block of cells
// requires rebuilding the flat array (interleaved columns don't grow cheaply).
static void merge_cells(Pipeline::CellData &dst, CellLocal &src)
{
    if (src.n() == 0) return;
    const int64_t off   = dst.n;
    const int64_t new_n = off + src.n();
    dst.n = new_n;

    vec_append(dst.x,     src.x);
    vec_append(dst.y,     src.y);
    vec_append(dst.z,     src.z);
    vec_append(dst.dx,    src.dx);
    vec_append(dst.level, src.level);

    const int32_t nv = src.nvar;
    if (dst.nvar == 0) dst.nvar = nv;

    // Rebuild flat column-major array
    std::vector<double> new_var((size_t)nv * (size_t)new_n);
    for (int32_t v = 0; v < nv; ++v) {
        const size_t col_base = (size_t)v * (size_t)new_n;
        if (off > 0) {
            std::copy(dst.var.begin() + (ptrdiff_t)((size_t)v * (size_t)off),
                      dst.var.begin() + (ptrdiff_t)((size_t)v * (size_t)off + (size_t)off),
                      new_var.begin() + (ptrdiff_t)col_base);
        }
        std::copy(src.var[(size_t)v].begin(), src.var[(size_t)v].end(),
                  new_var.begin() + (ptrdiff_t)(col_base + (size_t)off));
    }
    dst.var = std::move(new_var);
}

// ============================================================
// Old format — particle reading
// ============================================================

static bool read_part_domain_old(const std::string &fn, StarLocal &star, DMLocal &dm)
{
    FILE *fp = fopen(fn.c_str(), "rb");
    if (!fp) return false;

    // Header records: rec1=ncpu, rec2=ndim, rec3=npart, rec4-rec8 skip
    frec_skip(fp); frec_skip(fp);
    int32_t npart = 0;
    frec_read(fp, &npart, 1);
    frec_skip(fp); frec_skip(fp); frec_skip(fp); frec_skip(fp); frec_skip(fp);

    if (npart <= 0) { fclose(fp); return true; }

    std::vector<double>  xp(npart), yp(npart), zp(npart);
    std::vector<double>  vxp(npart), vyp(npart), vzp(npart);
    std::vector<double>  mp(npart);
    std::vector<int64_t> idp(npart);
    std::vector<int8_t>  fam(npart, 0);
    std::vector<double>  agep(npart, 0.0), metal(npart, 0.0);

    frec_read(fp, xp.data(), npart);
    frec_read(fp, yp.data(), npart);
    frec_read(fp, zp.data(), npart);
    frec_read(fp, vxp.data(), npart);
    frec_read(fp, vyp.data(), npart);
    frec_read(fp, vzp.data(), npart);
    frec_read(fp, mp.data(), npart);

    // ID: auto-detect int32 vs int64 from record byte-length
    {
        const int32_t id_len = frec_peek_len(fp);
        if (id_len == npart * 8) {
            frec_read(fp, idp.data(), npart);
        } else {
            std::vector<int32_t> tmp(npart);
            frec_read(fp, tmp.data(), npart);
            for (int32_t i = 0; i < npart; ++i) idp[i] = (int64_t)tmp[i];
        }
    }

    frec_skip(fp);  // levelp

    // Family: stored as uint8; values > 100 are negative signed (subtract 255)
    {
        int32_t len;
        fread(&len, 4, 1, fp);
        std::vector<uint8_t> fu((size_t)npart);
        fread(fu.data(), 1, (size_t)npart, fp);
        fread(&len, 4, 1, fp);
        for (int32_t i = 0; i < npart; ++i)
            fam[i] = (fu[i] > 100) ? (int8_t)((int)fu[i] - 255) : (int8_t)fu[i];
    }

    frec_skip(fp);  // tag

    bool has_age = (frec_peek_len(fp) > 0);
    if (has_age)   has_age   = frec_read(fp, agep.data(), npart);
    bool has_metal = has_age && (frec_peek_len(fp) > 0);
    if (has_metal) frec_read(fp, metal.data(), npart);

    fclose(fp);

    for (int32_t i = 0; i < npart; ++i) {
        if (fam[i] == 2) {
            star.x.push_back(xp[i]);   star.y.push_back(yp[i]);   star.z.push_back(zp[i]);
            star.vx.push_back((float)vxp[i]); star.vy.push_back((float)vyp[i]);
            star.vz.push_back((float)vzp[i]); star.mass.push_back((float)mp[i]);
            star.id.push_back(idp[i]);
            star.age.push_back(has_age   ? (float)agep[i]  : 0.0f);
            star.metal.push_back(has_metal ? (float)metal[i] : 0.0f);
        } else if (fam[i] == 1) {
            dm.x.push_back(xp[i]);   dm.y.push_back(yp[i]);   dm.z.push_back(zp[i]);
            dm.vx.push_back((float)vxp[i]); dm.vy.push_back((float)vyp[i]);
            dm.vz.push_back((float)vzp[i]); dm.mass.push_back((float)mp[i]);
            dm.id.push_back(idp[i]);
        }
    }
    return true;
}

// ============================================================
// Old format — AMR cell reading
// ============================================================

struct CellParams {
    int32_t ncpu      = 0;
    int32_t ndim      = 3;
    int32_t levelmax  = 0;
    int32_t nboundary = 0;
    int32_t nvarh     = 0;
    int32_t twotondim = 8;
};

// Read global grid parameters from CPU-1's AMR + hydro files.
static bool read_cell_params_old(const std::string &sd, int32_t snap, CellParams &p)
{
    // Each header value is its own Fortran record:
    // ncpu | ndim | nx,ny,nz | nlevelmax | ngridmax | nboundary
    FILE *fa = fopen(fname_amr(sd, snap, 1).c_str(), "rb");
    if (!fa) return false;
    int32_t ncpu = 0, ndim = 0, nxyz[3] = {}, lmax = 0, ngmax = 0, nb = 0;
    if (!frec_read(fa, &ncpu, 1) || !frec_read(fa, &ndim, 1) ||
        !frec_read(fa, nxyz, 3)  || !frec_read(fa, &lmax, 1) ||
        !frec_read(fa, &ngmax, 1) || !frec_read(fa, &nb, 1)) {
        fclose(fa); return false;
    }
    fclose(fa);
    p.ncpu      = ncpu;
    p.ndim      = ndim;
    p.levelmax  = lmax;
    p.nboundary = nb;
    p.twotondim = 1 << p.ndim;

    // Hydro: rec1 = ncpu, rec2 = nvarh
    FILE *fh = fopen(fname_hydro(sd, snap, 1).c_str(), "rb");
    if (!fh) return false;
    int32_t hncpu = 0;
    if (!frec_read(fh, &hncpu, 1) || !frec_read(fh, &p.nvarh, 1)) {
        fclose(fh); return false;
    }
    fclose(fh);

    return (p.ncpu > 0 && p.ndim >= 1 && p.levelmax > 0 && p.nvarh > 0);
}

// Read all leaf cells owned by icpu from its AMR + hydro binary files.
// Mirrors the octree traversal algorithm in jsamr2cell.f90.
//
// AMR file structure after the 21-record header:
//   ngrid[ndomain][levelmax]  one packed Fortran record
//   1 skip
//   (if nboundary>0): 2 skip + ngrid_b record
//   2 skip + ordering + 3 skip
//   Level loop for each (level, j in 0..ndomain-1):
//     When ngrida > 0:
//       AMR block: 3 preamble + ndim xg + father + 2*ndim nbors
//                  + twotondim son + twotondim cpu_map + twotondim refmap
//     Hydro block (always 2 records); when ngrida > 0: twotondim*nvarh data records
//
// Leaf cell condition: son[ind][k] == 0.
// Cell position: xg[dim][k] + xc[ind][dim],  xc = child offset for index ind.
//
static bool read_cell_domain_old(const std::string &fa_name,
                                  const std::string &fh_name,
                                  int32_t icpu,
                                  const CellParams &p,
                                  CellLocal &local)
{
    FILE *fa = fopen(fa_name.c_str(), "rb");
    if (!fa) return false;
    FILE *fh = fopen(fh_name.c_str(), "rb");
    if (!fh) { fclose(fa); return false; }

    const int32_t ndomain   = p.ncpu + p.nboundary;
    const int32_t twotondim = p.twotondim;
    const int32_t ndim      = p.ndim;
    const int32_t levelmax  = p.levelmax;
    const int32_t nvarh     = p.nvarh;

    // --- AMR header: skip 21 records ---
    for (int i = 0; i < 21; ++i) frec_skip(fa);

    // ngrid[j][lev]  Fortran column-major: stride = ndomain
    const int32_t ng_total = ndomain * levelmax;
    std::vector<int32_t> ngrid((size_t)ng_total, 0);
    frec_read(fa, ngrid.data(), ng_total);
    auto get_ng = [&](int32_t j0, int32_t lev0) -> int32_t {
        return ngrid[(size_t)j0 + (size_t)lev0 * (size_t)ndomain];
    };

    frec_skip(fa);
    if (p.nboundary > 0) { frec_skip(fa); frec_skip(fa); frec_skip(fa); }
    frec_skip(fa); frec_skip(fa);
    frec_skip(fa);  // ordering
    frec_skip(fa); frec_skip(fa); frec_skip(fa);

    // --- Hydro header: 6 records ---
    for (int i = 0; i < 6; ++i) frec_skip(fh);

    local.init(nvarh);

    // Determine max octs per domain for buffer sizing
    int32_t max_ng = 0;
    for (int32_t lev = 0; lev < levelmax; ++lev)
        for (int32_t j = 0; j < ndomain; ++j)
            if (get_ng(j, lev) > max_ng) max_ng = get_ng(j, lev);

    if (max_ng == 0) { fclose(fa); fclose(fh); return true; }

    // Grid-centre coordinates (one dim at a time), son, hydro
    std::vector<double>  xg_tmp((size_t)max_ng);
    std::vector<double>  xg[3];
    for (int d = 0; d < ndim && d < 3; ++d) xg[d].resize((size_t)max_ng);
    std::vector<std::vector<int32_t>> son_arr((size_t)twotondim,
                                               std::vector<int32_t>((size_t)max_ng));
    std::vector<std::vector<std::vector<double>>> hd_arr(
        (size_t)twotondim,
        std::vector<std::vector<double>>((size_t)nvarh,
                                          std::vector<double>((size_t)max_ng)));
    std::vector<double> hd_cell((size_t)nvarh);

    // --- Level loop ---
    for (int32_t ilevel = 1; ilevel <= levelmax; ++ilevel) {
        const double dx  = std::pow(0.5, (double)ilevel);
        const float  fdx = (float)dx;

        // Child offsets within an oct: ind = ix + 2*iy + 4*iz
        double xc[8][3] = {};
        for (int ind = 0; ind < twotondim; ++ind) {
            const int iz = ind / 4;
            const int iy = (ind - 4 * iz) / 2;
            const int ix = ind - 2 * iy - 4 * iz;
            xc[ind][0] = ((double)ix - 0.5) * dx;
            xc[ind][1] = ((double)iy - 0.5) * dx;
            xc[ind][2] = (ndim >= 3) ? ((double)iz - 0.5) * dx : 0.0;
        }

        for (int32_t j = 0; j < ndomain; ++j) {
            const int32_t ngrida = get_ng(j, ilevel - 1);
            const bool    own    = (j == icpu - 1);

            // --- AMR block (only when ngrida > 0) ---
            if (ngrida > 0) {
                frec_skip(fa); frec_skip(fa); frec_skip(fa);  // 3 preamble

                if (own) {
                    for (int d = 0; d < ndim && d < 3; ++d) {
                        frec_read(fa, xg_tmp.data(), ngrida);
                        std::copy(xg_tmp.begin(), xg_tmp.begin() + ngrida, xg[d].begin());
                    }
                    if (ndim > 3) { for (int d = 3; d < ndim; ++d) frec_skip(fa); }
                } else {
                    for (int32_t d = 0; d < ndim; ++d) frec_skip(fa);
                }

                frec_skip(fa);  // father
                for (int32_t k = 0; k < 2 * ndim; ++k) frec_skip(fa);  // neighbors

                if (own) {
                    for (int32_t ind = 0; ind < twotondim; ++ind)
                        frec_read(fa, son_arr[(size_t)ind].data(), ngrida);
                } else {
                    for (int32_t ind = 0; ind < twotondim; ++ind) frec_skip(fa);
                }

                // twotondim cpu_map + twotondim refmap
                for (int32_t k = 0; k < 2 * twotondim; ++k) frec_skip(fa);
            }

            // --- Hydro block: 2 header records always ---
            frec_skip(fh); frec_skip(fh);

            if (ngrida > 0) {
                if (own) {
                    for (int32_t ind = 0; ind < twotondim; ++ind)
                        for (int32_t iv = 0; iv < nvarh; ++iv)
                            frec_read(fh, hd_arr[(size_t)ind][(size_t)iv].data(), ngrida);

                    // Extract leaf cells (son == 0 ↔ no child oct)
                    for (int32_t k = 0; k < ngrida; ++k) {
                        for (int32_t ind = 0; ind < twotondim; ++ind) {
                            if (son_arr[(size_t)ind][(size_t)k] == 0) {
                                double cx = xg[0][(size_t)k] + xc[ind][0];
                                double cy = (ndim >= 2) ? xg[1][(size_t)k] + xc[ind][1] : 0.0;
                                double cz = (ndim >= 3) ? xg[2][(size_t)k] + xc[ind][2] : 0.0;
                                for (int32_t v = 0; v < nvarh; ++v)
                                    hd_cell[(size_t)v] = hd_arr[(size_t)ind][(size_t)v][(size_t)k];
                                local.push(cx, cy, cz, fdx, ilevel, hd_cell);
                            }
                        }
                    }
                } else {
                    for (int32_t k = 0; k < twotondim * nvarh; ++k) frec_skip(fh);
                }
            }
        }
    }

    fclose(fa);
    fclose(fh);
    return true;
}

static void read_cells_old(const vpp_set::Settings &vh, int32_t snap,
                            int32_t n_cpu, Pipeline::RawData &raw)
{
    const std::string sd = snap_dir(vh, snap);

    CellParams p;
    if (!read_cell_params_old(sd, snap, p)) {
        LOG() << "  [read_cells_old] cannot read cell params";
        return;
    }
    LOG() << "  [read_cells_old] ncpu=" << p.ncpu
          << "  ndim=" << p.ndim << "  levmax=" << p.levelmax
          << "  nvarh=" << p.nvarh;

    int n_threads = 1;
#ifdef VPP_USE_OMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<CellLocal> cell_th((size_t)n_threads);

#ifdef VPP_USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int32_t i = 0; i < n_cpu; ++i) {
        int tid = 0;
#ifdef VPP_USE_OMP
        tid = omp_get_thread_num();
#endif
        const int32_t icpu = i + 1;
        read_cell_domain_old(fname_amr(sd, snap, icpu),
                             fname_hydro(sd, snap, icpu),
                             icpu, p, cell_th[(size_t)tid]);
    }

    for (int t = 0; t < n_threads; ++t)
        merge_cells(raw.cell, cell_th[(size_t)t]);

    LOG() << "  [read_cells_old] n_cell=" << raw.cell.n
          << "  nvar=" << raw.cell.nvar;
}

// ============================================================
// New HDF5 format
// ============================================================

#ifdef VPP_USE_HDF5

// Particle compound types (packed, matching read_ramseshdf_conly.c)
typedef struct {
    double  position_x, position_y, position_z;
    float   velocity_x, velocity_y, velocity_z;
    float   mass;
    int32_t identity;
    int8_t  levelp, family, tag;
    float   birth_time, metallicity, initial_mass, birth_density;
    int16_t cpu;
} __attribute__((packed)) star_record_t;

typedef struct {
    double  position_x, position_y, position_z;
    float   velocity_x, velocity_y, velocity_z;
    float   mass;
    int32_t identity;
    int8_t  levelp, family, tag;
    int16_t cpu;
} __attribute__((packed)) dm_record_t;

// Cell compound type (matches read_ramseshdf_conly.c cell_record_t)
typedef struct {
    double position_x, position_y, position_z;
    int    level;
    float  density, velocity_x, velocity_y, velocity_z,
           pressure, metallicity, potential;
} cell_record_t;

static hid_t make_star_mt()
{
    hid_t mt = H5Tcreate(H5T_COMPOUND, sizeof(star_record_t));
    H5Tinsert(mt, "position_x",    HOFFSET(star_record_t, position_x),    H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "position_y",    HOFFSET(star_record_t, position_y),    H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "position_z",    HOFFSET(star_record_t, position_z),    H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "velocity_x",    HOFFSET(star_record_t, velocity_x),    H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "velocity_y",    HOFFSET(star_record_t, velocity_y),    H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "velocity_z",    HOFFSET(star_record_t, velocity_z),    H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "mass",          HOFFSET(star_record_t, mass),          H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "identity",      HOFFSET(star_record_t, identity),      H5T_NATIVE_INT);
    H5Tinsert(mt, "levelp",        HOFFSET(star_record_t, levelp),        H5T_NATIVE_INT8);
    H5Tinsert(mt, "family",        HOFFSET(star_record_t, family),        H5T_NATIVE_INT8);
    H5Tinsert(mt, "tag",           HOFFSET(star_record_t, tag),           H5T_NATIVE_INT8);
    H5Tinsert(mt, "birth_time",    HOFFSET(star_record_t, birth_time),    H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "metallicity",   HOFFSET(star_record_t, metallicity),   H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "initial_mass",  HOFFSET(star_record_t, initial_mass),  H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "birth_density", HOFFSET(star_record_t, birth_density), H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "cpu",           HOFFSET(star_record_t, cpu),           H5T_NATIVE_SHORT);
    return mt;
}

static hid_t make_dm_mt()
{
    hid_t mt = H5Tcreate(H5T_COMPOUND, sizeof(dm_record_t));
    H5Tinsert(mt, "position_x", HOFFSET(dm_record_t, position_x), H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "position_y", HOFFSET(dm_record_t, position_y), H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "position_z", HOFFSET(dm_record_t, position_z), H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "velocity_x", HOFFSET(dm_record_t, velocity_x), H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "velocity_y", HOFFSET(dm_record_t, velocity_y), H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "velocity_z", HOFFSET(dm_record_t, velocity_z), H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "mass",       HOFFSET(dm_record_t, mass),       H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "identity",   HOFFSET(dm_record_t, identity),   H5T_NATIVE_INT);
    H5Tinsert(mt, "levelp",     HOFFSET(dm_record_t, levelp),     H5T_NATIVE_INT8);
    H5Tinsert(mt, "family",     HOFFSET(dm_record_t, family),     H5T_NATIVE_INT8);
    H5Tinsert(mt, "tag",        HOFFSET(dm_record_t, tag),        H5T_NATIVE_INT8);
    H5Tinsert(mt, "cpu",        HOFFSET(dm_record_t, cpu),        H5T_NATIVE_SHORT);
    return mt;
}

static hid_t make_cell_mt()
{
    hid_t mt = H5Tcreate(H5T_COMPOUND, sizeof(cell_record_t));
    H5Tinsert(mt, "position_x",  HOFFSET(cell_record_t, position_x),  H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "position_y",  HOFFSET(cell_record_t, position_y),  H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "position_z",  HOFFSET(cell_record_t, position_z),  H5T_NATIVE_DOUBLE);
    H5Tinsert(mt, "level",       HOFFSET(cell_record_t, level),       H5T_NATIVE_INT);
    H5Tinsert(mt, "density",     HOFFSET(cell_record_t, density),     H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "velocity_x",  HOFFSET(cell_record_t, velocity_x),  H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "velocity_y",  HOFFSET(cell_record_t, velocity_y),  H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "velocity_z",  HOFFSET(cell_record_t, velocity_z),  H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "pressure",    HOFFSET(cell_record_t, pressure),    H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "metallicity", HOFFSET(cell_record_t, metallicity), H5T_NATIVE_FLOAT);
    H5Tinsert(mt, "potential",   HOFFSET(cell_record_t, potential),   H5T_NATIVE_FLOAT);
    return mt;
}

// Shared batch+parallel reader used for both star and dm groups.
// Opens nthreads independent file handles so LZF decompresses in parallel.
template <typename RecT, typename MtFn, typename ScatterFn>
static void read_ptcl_group_batched(const std::string &fn,
                                    const char *grpname,
                                    int nthreads,
                                    hsize_t n,
                                    MtFn    make_mt,
                                    ScatterFn scatter)
{
    std::vector<hid_t> t_fid(nthreads), t_gid(nthreads), t_did(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        t_fid[t] = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        t_gid[t] = H5Gopen(t_fid[t], grpname, H5P_DEFAULT);
        t_did[t] = H5Dopen(t_gid[t], "data",  H5P_DEFAULT);
    }

    const hsize_t BATCH   = 4000000;
    const hsize_t n_batch = (n + BATCH - 1) / BATCH;
    const auto t0 = std::chrono::steady_clock::now();

    for (hsize_t b = 0; b < n_batch; ++b) {
        const hsize_t b_start = b * BATCH;
        const hsize_t b_count = std::min(BATCH, n - b_start);
        const hsize_t t_chunk = (b_count + (hsize_t)nthreads - 1) / (hsize_t)nthreads;

#ifdef VPP_USE_OMP
#pragma omp parallel num_threads(nthreads)
        {
            const int     tid    = omp_get_thread_num();
            const hsize_t t_off  = (hsize_t)tid * t_chunk;
            if (t_off < b_count) {
                const hsize_t t_count = std::min(t_chunk, b_count - t_off);
                const hsize_t g_start = b_start + t_off;
                std::vector<RecT> tbuf((size_t)t_count);
                hid_t mt   = make_mt();
                hid_t fsid = H5Dget_space(t_did[tid]);
                H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &g_start, nullptr, &t_count, nullptr);
                hid_t msid = H5Screate_simple(1, &t_count, nullptr);
                H5Dread(t_did[tid], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
                H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);
                for (hsize_t i = 0; i < t_count; ++i)
                    scatter(tbuf[i], g_start + i);
            }
        }
#else
        {
            std::vector<RecT> tbuf((size_t)b_count);
            hid_t mt   = make_mt();
            hid_t fsid = H5Dget_space(t_did[0]);
            H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &b_start, nullptr, &b_count, nullptr);
            hid_t msid = H5Screate_simple(1, &b_count, nullptr);
            H5Dread(t_did[0], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
            H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);
            for (hsize_t i = 0; i < b_count; ++i)
                scatter(tbuf[i], b_start + i);
        }
#endif
        if (b % 50 == 49 || b == n_batch - 1) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            LOG() << "    [read_particles_hdf] " << grpname
                  << " batch " << (b+1) << "/" << n_batch
                  << "  " << std::fixed << std::setprecision(1)
                  << 100.0*(double)(b+1)/(double)n_batch << "%"
                  << "  elapsed=" << elapsed << "s";
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        H5Dclose(t_did[t]); H5Gclose(t_gid[t]); H5Fclose(t_fid[t]);
    }
}

// Cell HDF5 file: {dir_raw}/../hdf/cell_{snap:05d}.h5
// Group "leaf", dataset "data" of compound type cell_record_t.
// nvar = 7: density, vx, vy, vz, pressure (rho*T/mu), metallicity, potential
//
// Performance notes (snap_00030: 1B cells, 27-field/118B compound, LZF+shuffle):
//   Old: single 56 GB intermediate buffer + single-threaded LZF decompress
//   New: 4 M-cell batches × per-thread H5Fopen → parallel LZF decompress,
//        peak buffer ~(4M × 56B × nthreads) ≈ 1.8 GB @ 8 threads
static void read_cells_hdf(const vpp_set::Settings &vh, int32_t snap,
                            Pipeline::RawData &raw)
{
    std::ostringstream o;
    o << vh.dir_raw << "/../hdf/cell_"
      << std::setw(5) << std::setfill('0') << snap << ".h5";
    const std::string fn = o.str();

    // Probe file once to get cell count
    hsize_t nc = 0;
    {
        hid_t fid0 = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid0 < 0) { LOG() << "  [read_cells_hdf] file not found: " << fn; return; }
        if (H5Lexists(fid0, "leaf", H5P_DEFAULT) <= 0) {
            LOG() << "  [read_cells_hdf] group 'leaf' not found";
            H5Fclose(fid0); return;
        }
        hid_t gid0 = H5Gopen(fid0, "leaf", H5P_DEFAULT);
        if (H5Lexists(gid0, "data", H5P_DEFAULT) <= 0) {
            LOG() << "  [read_cells_hdf] dataset 'leaf/data' not found";
            H5Gclose(gid0); H5Fclose(fid0); return;
        }
        hid_t did0 = H5Dopen(gid0, "data", H5P_DEFAULT);
        hid_t sid0 = H5Dget_space(did0);
        hsize_t dims[1] = {0};
        H5Sget_simple_extent_dims(sid0, dims, nullptr);
        nc = dims[0];
        H5Sclose(sid0); H5Dclose(did0); H5Gclose(gid0); H5Fclose(fid0);
    }
    if (nc == 0) return;

    // Pre-allocate output (nvar=7 fixed: density,vx,vy,vz,pressure,metallicity,potential)
    const int32_t nvar = 7;
    raw.cell.n    = (int64_t)nc;
    raw.cell.nvar = nvar;
    raw.cell.x.resize((size_t)nc);
    raw.cell.y.resize((size_t)nc);
    raw.cell.z.resize((size_t)nc);
    raw.cell.dx.resize((size_t)nc);
    raw.cell.level.resize((size_t)nc);
    raw.cell.var.resize((size_t)nvar * (size_t)nc);

#ifdef VPP_USE_OMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif

    LOG() << "  [read_cells_hdf] " << nc << " cells  threads=" << nthreads;

    // Open one independent HDF5 handle per thread so LZF decompression runs in parallel
    std::vector<hid_t> t_fid(nthreads, -1);
    std::vector<hid_t> t_gid(nthreads, -1);
    std::vector<hid_t> t_did(nthreads, -1);
    for (int t = 0; t < nthreads; ++t) {
        t_fid[t] = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        t_gid[t] = H5Gopen(t_fid[t], "leaf", H5P_DEFAULT);
        t_did[t] = H5Dopen(t_gid[t], "data", H5P_DEFAULT);
    }

    // 4M cells per batch: per-thread buffer ~224 MB, total ~1.8 GB @ 8 threads
    const hsize_t BATCH   = 4000000;
    const hsize_t n_batch = (nc + BATCH - 1) / BATCH;

    using clk = std::chrono::steady_clock;
    const auto t_wall0 = clk::now();

    for (hsize_t b = 0; b < n_batch; ++b) {
        const hsize_t b_start = b * BATCH;
        const hsize_t b_count = std::min(BATCH, nc - b_start);
        const hsize_t t_chunk = (b_count + (hsize_t)nthreads - 1) / (hsize_t)nthreads;

#ifdef VPP_USE_OMP
#pragma omp parallel num_threads(nthreads)
        {
            const int     tid     = omp_get_thread_num();
            const hsize_t t_off   = (hsize_t)tid * t_chunk;
            if (t_off < b_count) {
                const hsize_t t_count = std::min(t_chunk, b_count - t_off);
                const hsize_t g_start = b_start + t_off;

                std::vector<cell_record_t> tbuf((size_t)t_count);
                hid_t mt   = make_cell_mt();
                hid_t fsid = H5Dget_space(t_did[tid]);
                H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &g_start, nullptr, &t_count, nullptr);
                hid_t msid = H5Screate_simple(1, &t_count, nullptr);
                H5Dread(t_did[tid], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
                H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);

                for (hsize_t i = 0; i < t_count; ++i) {
                    const size_t gi = (size_t)(g_start + i);
                    const cell_record_t &r = tbuf[i];
                    raw.cell.x[gi]                  = r.position_x;
                    raw.cell.y[gi]                  = r.position_y;
                    raw.cell.z[gi]                  = r.position_z;
                    raw.cell.level[gi]              = r.level;
                    raw.cell.dx[gi]                 = (float)(1.0 / (double)(1 << r.level));
                    raw.cell.var[0*(size_t)nc + gi] = (double)r.density;
                    raw.cell.var[1*(size_t)nc + gi] = (double)r.velocity_x;
                    raw.cell.var[2*(size_t)nc + gi] = (double)r.velocity_y;
                    raw.cell.var[3*(size_t)nc + gi] = (double)r.velocity_z;
                    raw.cell.var[4*(size_t)nc + gi] = (double)r.pressure;
                    raw.cell.var[5*(size_t)nc + gi] = (double)r.metallicity;
                    raw.cell.var[6*(size_t)nc + gi] = (double)r.potential;
                }
            }
        }
#else
        {
            std::vector<cell_record_t> tbuf((size_t)b_count);
            hid_t mt   = make_cell_mt();
            hid_t fsid = H5Dget_space(t_did[0]);
            H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &b_start, nullptr, &b_count, nullptr);
            hid_t msid = H5Screate_simple(1, &b_count, nullptr);
            H5Dread(t_did[0], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
            H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);
            for (hsize_t i = 0; i < b_count; ++i) {
                const size_t gi = (size_t)(b_start + i);
                const cell_record_t &r = tbuf[i];
                raw.cell.x[gi]                  = r.position_x;
                raw.cell.y[gi]                  = r.position_y;
                raw.cell.z[gi]                  = r.position_z;
                raw.cell.level[gi]              = r.level;
                raw.cell.dx[gi]                 = (float)(1.0 / (double)(1 << r.level));
                raw.cell.var[0*(size_t)nc + gi] = (double)r.density;
                raw.cell.var[1*(size_t)nc + gi] = (double)r.velocity_x;
                raw.cell.var[2*(size_t)nc + gi] = (double)r.velocity_y;
                raw.cell.var[3*(size_t)nc + gi] = (double)r.velocity_z;
                raw.cell.var[4*(size_t)nc + gi] = (double)r.pressure;
                raw.cell.var[5*(size_t)nc + gi] = (double)r.metallicity;
                raw.cell.var[6*(size_t)nc + gi] = (double)r.potential;
            }
        }
#endif
        // Progress log every 50 batches (~200M cells) and at the end
        if (b % 50 == 49 || b == n_batch - 1) {
            const double elapsed = std::chrono::duration<double>(clk::now() - t_wall0).count();
            const double pct     = 100.0 * (double)(b + 1) / (double)n_batch;
            LOG() << "    [read_cells_hdf] batch " << (b + 1) << "/" << n_batch
                  << "  " << std::fixed << std::setprecision(1) << pct << "%"
                  << "  elapsed=" << std::setprecision(1) << elapsed << "s";
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        H5Dclose(t_did[t]); H5Gclose(t_gid[t]); H5Fclose(t_fid[t]);
    }
}

#endif // VPP_USE_HDF5

// ---------------------------------------------------------------------------
// Streaming readers (public, HDF5 only)
// ---------------------------------------------------------------------------
#ifdef VPP_USE_HDF5

// Helper: open file+group+dataset handles for nthreads threads
static void open_handles(const std::string &fn, const char *grp,
                          int nthreads,
                          std::vector<hid_t> &t_fid,
                          std::vector<hid_t> &t_gid,
                          std::vector<hid_t> &t_did)
{
    t_fid.resize(nthreads); t_gid.resize(nthreads); t_did.resize(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        t_fid[t] = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        t_gid[t] = H5Gopen(t_fid[t], grp, H5P_DEFAULT);
        t_did[t] = H5Dopen(t_gid[t], "data", H5P_DEFAULT);
    }
}
static void close_handles(int nthreads,
                           std::vector<hid_t> &t_fid,
                           std::vector<hid_t> &t_gid,
                           std::vector<hid_t> &t_did)
{
    for (int t = 0; t < nthreads; ++t) {
        H5Dclose(t_did[t]); H5Gclose(t_gid[t]); H5Fclose(t_fid[t]);
    }
}

// Get the count of records in grp/data
static hsize_t hdf5_group_count(const std::string &fn, const char *grp)
{
    hid_t fid = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) return 0;
    if (H5Lexists(fid, grp, H5P_DEFAULT) <= 0) { H5Fclose(fid); return 0; }
    hid_t gid = H5Gopen(fid, grp, H5P_DEFAULT);
    if (H5Lexists(gid, "data", H5P_DEFAULT) <= 0) { H5Gclose(gid); H5Fclose(fid); return 0; }
    hid_t did = H5Dopen(gid, "data", H5P_DEFAULT);
    hid_t sid = H5Dget_space(did);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(sid, dims, nullptr);
    H5Sclose(sid); H5Dclose(did); H5Gclose(gid); H5Fclose(fid);
    return dims[0];
}

#endif // VPP_USE_HDF5

} // anonymous namespace

// ============================================================
// Public API
// ============================================================

namespace Pipeline {

RawData read_raw(const vpp_set::Settings &vh, int32_t snap)
{
    LOG() << "[read_raw] snap=" << snap
          << "  format=" << (vh.newramses == 0 ? "old-binary" : "HDF5");

    RawData raw;

    if (vh.newramses == 0) {
        const std::string sd = snap_dir(vh, snap);
        int32_t n_cpu = count_cpu_files(sd, snap);
        if (n_cpu == 0) {
            LOG() << "  WARNING: no part_*.out files in " << sd;
            return raw;
        }
        LOG() << "  n_cpu=" << n_cpu;
        // Skip particles — bprop/confrac use stream_ptcl_old directly.
        // Only cells are needed (for gasprop); skip if gasprop is disabled.
        if (!vh.skip_gasprop && !vh.gas_r.empty())
            read_cells_old(vh, snap, n_cpu, raw);
    } else {
#ifdef VPP_USE_HDF5
        // HDF5 path: skip star/dm loading — bprop/confrac use stream_*_hdf() directly.
        // Only cells are needed (for gasprop); skip if gasprop is disabled.
        if (!vh.skip_gasprop && !vh.gas_r.empty())
            read_cells_hdf(vh, snap, raw);
#else
        LOG() << "  ERROR: newramses=1 but built without HDF5 support";
        return raw;
#endif
    }

    LOG() << "  n_star=" << raw.star.n
          << "  n_dm="   << raw.dm.n
          << "  n_cell=" << raw.cell.n;

    return raw;
}

// Build k-d trees for star, dm, and leaf-cell positions.
// Uses OMP sections so all three trees are constructed in parallel
// (up to 3 threads; remaining threads sit idle during construction).
// When star or dm data was not loaded (HDF5 streaming path), those trees are skipped.
void build_kdtrees(RawData &raw)
{
    LOG() << "[build_kdtrees] building k-d trees";

#ifdef VPP_USE_OMP
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (raw.star.n > 0)
                raw.star_tree.build(raw.star.x.data(), raw.star.y.data(),
                                    raw.star.z.data(), raw.star.n);
        }
#pragma omp section
        {
            if (raw.dm.n > 0)
                raw.dm_tree.build(raw.dm.x.data(), raw.dm.y.data(),
                                  raw.dm.z.data(), raw.dm.n);
        }
#pragma omp section
        {
            raw.cell_tree.build(raw.cell.x.data(), raw.cell.y.data(),
                                raw.cell.z.data(), raw.cell.n);
        }
    }
#else
    if (raw.star.n > 0)
        raw.star_tree.build(raw.star.x.data(), raw.star.y.data(),
                            raw.star.z.data(), raw.star.n);
    if (raw.dm.n > 0)
        raw.dm_tree.build(raw.dm.x.data(), raw.dm.y.data(),
                          raw.dm.z.data(), raw.dm.n);
    raw.cell_tree.build(raw.cell.x.data(), raw.cell.y.data(),
                        raw.cell.z.data(), raw.cell.n);
#endif

    if (raw.star.n > 0)
        LOG() << "  star_tree: " << raw.star_tree.size() << " pts, "
              << raw.star_tree.nodes.size() << " nodes";
    if (raw.dm.n > 0)
        LOG() << "  dm_tree:   " << raw.dm_tree.size()   << " pts, "
              << raw.dm_tree.nodes.size()   << " nodes";
    LOG() << "  cell_tree: " << raw.cell_tree.size() << " pts, "
          << raw.cell_tree.nodes.size() << " nodes";
}

#ifdef VPP_USE_HDF5
void stream_stars_hdf(
    const std::string &part_fn, int nthreads,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const float *age, const float *metal,
                              const int64_t *id, int64_t n)> &cb)
{
    const hsize_t ns = hdf5_group_count(part_fn, "star");
    if (ns == 0) return;

    std::vector<hid_t> t_fid, t_gid, t_did;
    open_handles(part_fn, "star", nthreads, t_fid, t_gid, t_did);

    const hsize_t BATCH   = 4000000;
    const hsize_t n_batch = (ns + BATCH - 1) / BATCH;

    // Batch SoA buffers (owned here, reused across batches)
    std::vector<double>  bx, by, bz;
    std::vector<float>   bmass, bage, bmetal;
    std::vector<int64_t> bid;

    for (hsize_t b = 0; b < n_batch; ++b) {
        const hsize_t b_start = b * BATCH;
        const hsize_t b_count = std::min(BATCH, ns - b_start);
        const hsize_t t_chunk = (b_count + (hsize_t)nthreads - 1) / (hsize_t)nthreads;

        bx.resize(b_count); by.resize(b_count); bz.resize(b_count);
        bmass.resize(b_count); bage.resize(b_count); bmetal.resize(b_count);
        bid.resize(b_count);

#ifdef VPP_USE_OMP
#pragma omp parallel num_threads(nthreads)
        {
            const int     tid    = omp_get_thread_num();
            const hsize_t t_off  = (hsize_t)tid * t_chunk;
            if (t_off < b_count) {
                const hsize_t t_count = std::min(t_chunk, b_count - t_off);
                const hsize_t g_start = b_start + t_off;
                std::vector<star_record_t> tbuf((size_t)t_count);
                hid_t mt   = make_star_mt();
                hid_t fsid = H5Dget_space(t_did[tid]);
                H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &g_start, nullptr, &t_count, nullptr);
                hid_t msid = H5Screate_simple(1, &t_count, nullptr);
                H5Dread(t_did[tid], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
                H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);
                for (hsize_t i = 0; i < t_count; ++i) {
                    const size_t j = (size_t)(t_off + i);
                    bx[j]     = tbuf[i].position_x;
                    by[j]     = tbuf[i].position_y;
                    bz[j]     = tbuf[i].position_z;
                    bmass[j]  = tbuf[i].mass;
                    bage[j]   = tbuf[i].birth_time;
                    bmetal[j] = tbuf[i].metallicity;
                    bid[j]    = (int64_t)tbuf[i].identity;
                }
            }
        }
#else
        {
            std::vector<star_record_t> tbuf((size_t)b_count);
            hid_t mt   = make_star_mt();
            hid_t fsid = H5Dget_space(t_did[0]);
            H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &b_start, nullptr, &b_count, nullptr);
            hid_t msid = H5Screate_simple(1, &b_count, nullptr);
            H5Dread(t_did[0], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
            H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);
            for (hsize_t i = 0; i < b_count; ++i) {
                bx[i] = tbuf[i].position_x; by[i] = tbuf[i].position_y; bz[i] = tbuf[i].position_z;
                bmass[i] = tbuf[i].mass; bage[i] = tbuf[i].birth_time; bmetal[i] = tbuf[i].metallicity;
                bid[i] = (int64_t)tbuf[i].identity;
            }
        }
#endif
        if (!cb(bx.data(), by.data(), bz.data(),
                bmass.data(), bage.data(), bmetal.data(),
                bid.data(), (int64_t)b_count)) break;
    }

    close_handles(nthreads, t_fid, t_gid, t_did);
}

void stream_dm_hdf(
    const std::string &part_fn, int nthreads,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const int64_t *id, int64_t n)> &cb)
{
    const hsize_t nd = hdf5_group_count(part_fn, "dm");
    if (nd == 0) return;

    std::vector<hid_t> t_fid, t_gid, t_did;
    open_handles(part_fn, "dm", nthreads, t_fid, t_gid, t_did);

    const hsize_t BATCH   = 4000000;
    const hsize_t n_batch = (nd + BATCH - 1) / BATCH;

    std::vector<double>  bx, by, bz;
    std::vector<float>   bmass;
    std::vector<int64_t> bid;

    for (hsize_t b = 0; b < n_batch; ++b) {
        const hsize_t b_start = b * BATCH;
        const hsize_t b_count = std::min(BATCH, nd - b_start);
        const hsize_t t_chunk = (b_count + (hsize_t)nthreads - 1) / (hsize_t)nthreads;

        bx.resize(b_count); by.resize(b_count); bz.resize(b_count);
        bmass.resize(b_count); bid.resize(b_count);

#ifdef VPP_USE_OMP
#pragma omp parallel num_threads(nthreads)
        {
            const int     tid    = omp_get_thread_num();
            const hsize_t t_off  = (hsize_t)tid * t_chunk;
            if (t_off < b_count) {
                const hsize_t t_count = std::min(t_chunk, b_count - t_off);
                const hsize_t g_start = b_start + t_off;
                std::vector<dm_record_t> tbuf((size_t)t_count);
                hid_t mt   = make_dm_mt();
                hid_t fsid = H5Dget_space(t_did[tid]);
                H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &g_start, nullptr, &t_count, nullptr);
                hid_t msid = H5Screate_simple(1, &t_count, nullptr);
                H5Dread(t_did[tid], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
                H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);
                for (hsize_t i = 0; i < t_count; ++i) {
                    const size_t j = (size_t)(t_off + i);
                    bx[j]    = tbuf[i].position_x;
                    by[j]    = tbuf[i].position_y;
                    bz[j]    = tbuf[i].position_z;
                    bmass[j] = tbuf[i].mass;
                    bid[j]   = (int64_t)tbuf[i].identity;
                }
            }
        }
#else
        {
            std::vector<dm_record_t> tbuf((size_t)b_count);
            hid_t mt   = make_dm_mt();
            hid_t fsid = H5Dget_space(t_did[0]);
            H5Sselect_hyperslab(fsid, H5S_SELECT_SET, &b_start, nullptr, &b_count, nullptr);
            hid_t msid = H5Screate_simple(1, &b_count, nullptr);
            H5Dread(t_did[0], mt, msid, fsid, H5P_DEFAULT, tbuf.data());
            H5Sclose(msid); H5Sclose(fsid); H5Tclose(mt);
            for (hsize_t i = 0; i < b_count; ++i) {
                bx[i] = tbuf[i].position_x; by[i] = tbuf[i].position_y; bz[i] = tbuf[i].position_z;
                bmass[i] = tbuf[i].mass; bid[i] = (int64_t)tbuf[i].identity;
            }
        }
#endif
        if (!cb(bx.data(), by.data(), bz.data(),
                bmass.data(), bid.data(), (int64_t)b_count)) break;
    }

    close_handles(nthreads, t_fid, t_gid, t_did);
}
#endif // VPP_USE_HDF5

// ============================================================
// stream_ptcl_old — streaming for old Fortran binary format
// ============================================================
void stream_ptcl_old(
    const vpp_set::Settings &vh,
    int32_t snap,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const float *age, const float *metal,
                              const int64_t *id, int64_t n)> &star_cb,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const int64_t *id, int64_t n)> &dm_cb)
{
    const std::string sd = snap_dir(vh, snap);
    int32_t n_cpu = count_cpu_files(sd, snap);
    if (n_cpu == 0) {
        LOG() << "[stream_ptcl_old] WARNING: no CPU files found in " << sd;
        return;
    }
    LOG() << "[stream_ptcl_old] n_cpu=" << n_cpu;

    bool stop_star = false, stop_dm = false;
    for (int32_t i = 0; i < n_cpu; ++i) {
        if (stop_star && stop_dm) break;
        StarLocal star_loc;
        DMLocal   dm_loc;
        read_part_domain_old(fname_part(sd, snap, i + 1), star_loc, dm_loc);

        if (!stop_star && star_cb && !star_loc.id.empty())
            stop_star = !star_cb(star_loc.x.data(), star_loc.y.data(), star_loc.z.data(),
                                 star_loc.mass.data(), star_loc.age.data(), star_loc.metal.data(),
                                 star_loc.id.data(), (int64_t)star_loc.id.size());

        if (!stop_dm && dm_cb && !dm_loc.id.empty())
            stop_dm = !dm_cb(dm_loc.x.data(), dm_loc.y.data(), dm_loc.z.data(),
                             dm_loc.mass.data(), dm_loc.id.data(), (int64_t)dm_loc.id.size());
    }
    LOG() << "[stream_ptcl_old] done";
}

} // namespace Pipeline
