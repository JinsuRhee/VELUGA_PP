// gasprop.cxx  -- gas mass + pressure via Hilbert-chunked cell streaming
//
// newramses=1 (HDF5):
//   Read leaf/hilbert_boundary + leaf/chunk_boundary from cell_XXXXX.h5.
//   Per galaxy: find overlapping HDF5 chunks via Hilbert key query.
//   Stream only needed chunks; OMP-parallel distance filter per chunk.
//   Early galaxy finalization: finalize + free cell data when last needed
//   chunk is processed.
//
// newramses=0 (old Ramses binary):
//   Read hindex (Hilbert domain boundaries per CPU) from info_XXXXX.txt.
//   Per galaxy: find overlapping CPU files via 8-corner Hilbert query.
//   Stream only needed CPU files; OMP-parallel distance filter per CPU.
//   Same early finalization logic.
//
// apply_pot_ref_and_classify uses volume-weighted mean potential at grav_r
// boundary shell as the PE reference.

#include "pipeline/gasprop.h"
#include "pipeline/rawdata.h"

#ifdef VPP_USE_HDF5
#include <hdf5.h>
#endif
#ifdef VPP_USE_OMP
#include <omp.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

namespace Pipeline {

namespace {

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------
static constexpr double kB_cgs    = 1.38062e-16;
static constexpr double mH_cgs    = 1.6600e-24;  // 1 amu (matches IDL)
static constexpr double Msun_cgs  = 1.989e33;
static constexpr double kpc_cm    = 3.0857e21;
static constexpr double gamma_gas = 5.0 / 3.0;
static constexpr double Zsun      = 0.02;
static constexpr double minZval   = 0.1 * Zsun;
static constexpr int    N_SHELL   = 50;
static constexpr double X_H       = 0.76;        // hydrogen mass fraction (IDL: X = 0.76)

// ---------------------------------------------------------------------------
// Per-cell quantities
// ---------------------------------------------------------------------------
struct CellPhys {
    double x_kpc, y_kpc, z_kpc;
    double d_kpc;
    double vol_kpc3;
    double rho_cgs, nH_cc;
    double vx_kms, vy_kms, vz_kms;
    double mass_msun;
    double metal;
    double T_K;
    double PE_kms2;
    double KE_kms2;
    double UE_kms2;
    double vdot;
    int    celltype;
    bool   cold;
};

// ---------------------------------------------------------------------------
// Compact HDF5 record (selective compound read)
// 3*8 + 4 + 7*4 = 24 + 4 + 28 = 56 bytes
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct CellRec {
    double  x, y, z;
    int32_t level;
    float   rho, vx, vy, vz, pres, metal, pot;
};
#pragma pack(pop)
static_assert(sizeof(CellRec) == 56, "CellRec size mismatch");

// ---------------------------------------------------------------------------
// Old Ramses per-cell buffer (up to 6 hydro variables)
// ---------------------------------------------------------------------------
struct OldCellRec {
    double  x, y, z;
    double  dx;
    int32_t level;
    int32_t nvar;
    double  var[6];  // rho, px, py, pz, etot, metal (in code units)
};

// ---------------------------------------------------------------------------
// Hilbert3D state machine (from IDL veluga::g_hilbert3d / Fortran find_hilbert)
// Identical between HDF5 and old Ramses formats.
// ---------------------------------------------------------------------------
static const int8_t nstate_tbl[12][8] = {
    { 1, 2, 3, 2, 4, 5, 3, 5},
    { 2, 6, 0, 7, 8, 8, 0, 7},
    { 0, 9,10, 9, 1, 1,11,11},
    { 6, 0, 6,11, 9, 0, 9, 8},
    {11,11, 0, 7, 5, 9, 0, 7},
    { 4, 4, 8, 8, 0, 6,10, 6},
    { 5, 7, 5, 3, 1, 1,11,11},
    { 6, 1, 6,10, 9, 4, 9,10},
    {10, 3, 1, 1,10, 3, 5, 9},
    { 4, 4, 8, 8, 2, 7, 2, 3},
    { 7, 2,11, 2, 7, 5, 8, 5},
    {10, 3, 2, 6,10, 3, 4, 4},
};
static const int8_t hdigit_tbl[12][8] = {
    {0, 1, 3, 2, 7, 6, 4, 5},
    {0, 7, 1, 6, 3, 4, 2, 5},
    {0, 3, 7, 4, 1, 2, 6, 5},
    {2, 3, 1, 0, 5, 4, 6, 7},
    {4, 3, 5, 2, 7, 0, 6, 1},
    {6, 5, 1, 2, 7, 4, 0, 3},
    {4, 7, 3, 0, 5, 6, 2, 1},
    {6, 7, 5, 4, 1, 0, 2, 3},
    {2, 5, 3, 4, 1, 6, 0, 7},
    {2, 1, 5, 6, 3, 0, 4, 7},
    {4, 5, 7, 6, 3, 2, 0, 1},
    {6, 1, 7, 0, 5, 2, 4, 3},
};

// Returns Hilbert key in [0, 2^(3*bit_length)).
// Works for both HDF5 (uint64 key space) and old Ramses (scaled separately).
static uint64_t hilbert3d(int64_t ix, int64_t iy, int64_t iz, int32_t bit_length)
{
    uint64_t key = 0;
    int cstate   = 0;
    for (int bit = bit_length - 1; bit >= 0; --bit) {
        int sd = (int)(((ix >> bit) & 1) << 2)
               | (int)(((iy >> bit) & 1) << 1)
               |  (int)((iz >> bit) & 1);
        int hd = hdigit_tbl[cstate][sd];
        cstate = nstate_tbl[cstate][sd];
        int bp = 3 * bit;
        key |= (uint64_t)((hd >> 2) & 1) << (bp + 2);
        key |= (uint64_t)((hd >> 1) & 1) << (bp + 1);
        key |= (uint64_t)( hd        & 1) <<  bp;
    }
    return key;
}

// ---------------------------------------------------------------------------
// HDF5 chunk selection
// ---------------------------------------------------------------------------
static std::vector<int32_t> find_needed_chunks(
    double xc, double yc, double zc, double r_code,
    const std::vector<uint64_t> &hb_lo,
    int32_t levmax)
{
    double xmin = std::max(0.0, xc - r_code), xmax = std::min(1.0, xc + r_code);
    double ymin = std::max(0.0, yc - r_code), ymax = std::min(1.0, yc + r_code);
    double zmin = std::max(0.0, zc - r_code), zmax = std::min(1.0, zc + r_code);
    double box_side = std::min({xmax-xmin, ymax-ymin, zmax-zmin});
    if (box_side <= 0.0) return {};

    int32_t ld = (int32_t)std::ceil(-std::log(box_side) / std::log(2.0)) + 2;
    ld = std::max(1, std::min(ld, levmax));

    int64_t gsize = int64_t(1) << ld;
    double  gs    = 1.0 / (double)gsize;
    int32_t shift = 3 * (levmax - ld);

    int64_t ix0 = std::max(int64_t(0), (int64_t)std::floor(xmin/gs));
    int64_t ix1 = std::min(gsize-1,    (int64_t)std::floor(xmax/gs));
    int64_t iy0 = std::max(int64_t(0), (int64_t)std::floor(ymin/gs));
    int64_t iy1 = std::min(gsize-1,    (int64_t)std::floor(ymax/gs));
    int64_t iz0 = std::max(int64_t(0), (int64_t)std::floor(zmin/gs));
    int64_t iz1 = std::min(gsize-1,    (int64_t)std::floor(zmax/gs));

    if ((ix1-ix0+1)*(iy1-iy0+1)*(iz1-iz0+1) > 50000) {
        ld = std::max(1, ld - 2);
        gsize = int64_t(1) << ld; gs = 1.0/(double)gsize;
        shift = 3*(levmax - ld);
        ix0=std::max(int64_t(0),(int64_t)std::floor(xmin/gs)); ix1=std::min(gsize-1,(int64_t)std::floor(xmax/gs));
        iy0=std::max(int64_t(0),(int64_t)std::floor(ymin/gs)); iy1=std::min(gsize-1,(int64_t)std::floor(ymax/gs));
        iz0=std::max(int64_t(0),(int64_t)std::floor(zmin/gs)); iz1=std::min(gsize-1,(int64_t)std::floor(zmax/gs));
    }

    std::set<int32_t> needed;
    for (int64_t ix = ix0; ix <= ix1; ++ix)
    for (int64_t iy = iy0; iy <= iy1; ++iy)
    for (int64_t iz = iz0; iz <= iz1; ++iz) {
        uint64_t key  = hilbert3d(ix, iy, iz, ld);
        uint64_t kmin = key << shift;
        uint64_t kmax = (key + 1ULL) << shift;
        auto it_a = std::upper_bound(hb_lo.begin(), hb_lo.end(), kmin);
        int32_t ci0 = (int32_t)std::max(int64_t(0), (int64_t)(it_a-hb_lo.begin())-1);
        auto it_b = std::lower_bound(hb_lo.begin(), hb_lo.end(), kmax);
        int32_t ci1 = (int32_t)std::min(int64_t(7999),(int64_t)(it_b-hb_lo.begin())-1);
        for (int32_t ci = ci0; ci <= ci1; ++ci) needed.insert(ci);
    }
    return std::vector<int32_t>(needed.begin(), needed.end());
}

// ---------------------------------------------------------------------------
// Old Ramses CPU selection (replicates find_domain.f90 8-corner approach)
// hindex is in [0, 2^(3*(levmax+1))] space (double).
// ---------------------------------------------------------------------------
static std::vector<int32_t> find_needed_cpus_old(
    double xc, double yc, double zc, double r_code,
    const std::vector<double> &hindex_lo,
    const std::vector<double> &hindex_hi,
    int32_t levmax)
{
    int32_t ncpu = (int32_t)hindex_lo.size();
    if (r_code <= 0.0 || ncpu == 0) return {};

    double dmax = r_code;

    // bit_length: first j where 0.5^j < dmax → lmin = j, bit_length = j-1
    int lmin = 1;
    while (std::pow(0.5, (double)lmin) >= dmax) ++lmin;
    int bit_length = lmin - 1;

    if (bit_length <= 0) {
        // sphere covers more than half the box → include all CPUs
        std::vector<int32_t> all(ncpu);
        for (int32_t i = 0; i < ncpu; ++i) all[i] = i;
        return all;
    }

    int64_t maxdom = int64_t(1) << bit_length;
    // dkey = (2^(levmax+1) / maxdom)^3 — scales Hilbert key to hindex space
    double scale = (double)(int64_t(1) << (levmax + 1)) / (double)maxdom;
    double dkey  = scale * scale * scale;

    auto clamp_idx = [&](int64_t v) -> int64_t {
        return std::max(int64_t(0), std::min(maxdom - 1, v));
    };
    int64_t imin = clamp_idx((int64_t)std::floor((xc - dmax) * (double)maxdom));
    int64_t imax = clamp_idx((int64_t)std::floor((xc + dmax) * (double)maxdom));
    int64_t jmin = clamp_idx((int64_t)std::floor((yc - dmax) * (double)maxdom));
    int64_t jmax = clamp_idx((int64_t)std::floor((yc + dmax) * (double)maxdom));
    int64_t kmin = clamp_idx((int64_t)std::floor((zc - dmax) * (double)maxdom));
    int64_t kmax = clamp_idx((int64_t)std::floor((zc + dmax) * (double)maxdom));

    double bnd_min = 1e100, bnd_max = -1e100;
    for (int64_t ix : {imin, imax})
    for (int64_t iy : {jmin, jmax})
    for (int64_t iz : {kmin, kmax}) {
        uint64_t order = hilbert3d(ix, iy, iz, bit_length);
        double key_lo  = (double)order * dkey;
        double key_hi  = ((double)order + 1.0) * dkey;
        if (key_lo < bnd_min) bnd_min = key_lo;
        if (key_hi > bnd_max) bnd_max = key_hi;
    }

    std::vector<int32_t> needed;
    for (int32_t i = 0; i < ncpu; ++i)
        if (hindex_lo[i] < bnd_max && hindex_hi[i] > bnd_min)
            needed.push_back(i);
    return needed;
}

// ---------------------------------------------------------------------------
// Read hindex (Hilbert domain boundaries) from info_XXXXX.txt.
// Format after 20 header lines: one line per CPU with 3 values (id, lo, hi).
// ---------------------------------------------------------------------------
static bool read_hindex_old(
    const std::string &dir_raw, int32_t snap,
    int32_t ncpu,
    std::vector<double> &hindex_lo,
    std::vector<double> &hindex_hi)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/output_%05d/info_%05d.txt", snap, snap);
    const std::string path = dir_raw + buf;

    FILE *f = fopen(path.c_str(), "r");
    if (!f) { LOG() << "[gasprop] ERROR: cannot open " << path; return false; }

    char line[512];
    for (int i = 0; i < 20; ++i) {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return false; }
    }

    hindex_lo.assign((size_t)ncpu, 0.0);
    hindex_hi.assign((size_t)ncpu, 0.0);
    int count = 0;
    while (count < ncpu) {
        if (!fgets(line, sizeof(line), f)) break;
        double a, b, c;
        if (sscanf(line, "%lf %lf %lf", &a, &b, &c) == 3) {
            // a = domain id (1-based), b = lo, c = hi
            hindex_lo[count] = b;
            hindex_hi[count] = c;
            ++count;
        }
        // Skip non-numeric lines (e.g. "domain from  to")
    }
    fclose(f);
    if (count != ncpu) {
        LOG() << "[gasprop] WARNING: hindex read " << count << "/" << ncpu;
        return count > 0;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fortran unformatted binary helpers (duplicated from rawdata.cxx)
// ---------------------------------------------------------------------------
static void frec_skip(FILE *fp)
{
    int32_t len;
    if (fread(&len, 4, 1, fp) != 1) return;
    fseek(fp, (long)len, SEEK_CUR);
    fread(&len, 4, 1, fp);
}

template <typename T>
static bool frec_read(FILE *fp, T *dst, int32_t n)
{
    int32_t len;
    if (fread(&len, 4, 1, fp) != 1) return false;
    fread(dst, sizeof(T), (size_t)n, fp);
    fread(&len, 4, 1, fp);
    return true;
}

// Grid parameters from AMR + hydro files.
struct CellParamsOld {
    int32_t ncpu      = 0;
    int32_t ndim      = 3;
    int32_t levelmax  = 0;
    int32_t nboundary = 0;
    int32_t nvarh     = 0;
    int32_t twotondim = 8;
};

static bool read_cell_params_old(const std::string &sd,
                                  int32_t snap,
                                  CellParamsOld &p)
{
    auto amr_name = [&]() {
        std::ostringstream o;
        o << sd << "/amr_" << std::setw(5) << std::setfill('0') << snap
                 << ".out00001";
        return o.str();
    };
    auto hydro_name = [&]() {
        std::ostringstream o;
        o << sd << "/hydro_" << std::setw(5) << std::setfill('0') << snap
                  << ".out00001";
        return o.str();
    };

    FILE *fa = fopen(amr_name().c_str(), "rb");
    if (!fa) return false;
    // Each header value is its own Fortran record:
    // ncpu | ndim | nx,ny,nz | nlevelmax | ngridmax | nboundary
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
    FILE *fh = fopen(hydro_name().c_str(), "rb");
    if (!fh) return false;
    int32_t hncpu = 0;
    if (!frec_read(fh, &hncpu, 1) || !frec_read(fh, &p.nvarh, 1)) {
        fclose(fh); return false;
    }
    fclose(fh);

    return (p.ncpu > 0 && p.ndim >= 1 && p.levelmax > 0 && p.nvarh > 0);
}

// Read all leaf cells owned by icpu from its AMR+hydro binary files.
// Returns up to nvar_use=min(nvarh,6) hydro variables per cell.
static std::vector<OldCellRec> read_cpu_cells_old(
    const std::string &fa_name,
    const std::string &fh_name,
    int32_t icpu,
    const CellParamsOld &p)
{
    FILE *fa = fopen(fa_name.c_str(), "rb");
    if (!fa) return {};
    FILE *fh = fopen(fh_name.c_str(), "rb");
    if (!fh) { fclose(fa); return {}; }

    const int32_t ndomain   = p.ncpu + p.nboundary;
    const int32_t twotondim = p.twotondim;
    const int32_t ndim      = p.ndim;
    const int32_t levelmax  = p.levelmax;
    const int32_t nvarh     = p.nvarh;
    const int32_t nvar_use  = std::min(nvarh, 6);

    for (int i = 0; i < 21; ++i) frec_skip(fa);

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

    for (int i = 0; i < 6; ++i) frec_skip(fh);

    int32_t max_ng = 0;
    for (int32_t lev = 0; lev < levelmax; ++lev)
        for (int32_t j = 0; j < ndomain; ++j)
            if (get_ng(j, lev) > max_ng) max_ng = get_ng(j, lev);

    if (max_ng == 0) { fclose(fa); fclose(fh); return {}; }

    std::vector<double> xg_tmp((size_t)max_ng);
    std::vector<double> xg[3];
    for (int d = 0; d < 3; ++d) xg[d].resize((size_t)max_ng);
    std::vector<std::vector<int32_t>> son_arr(
        (size_t)twotondim, std::vector<int32_t>((size_t)max_ng));
    std::vector<std::vector<std::vector<double>>> hd_arr(
        (size_t)twotondim,
        std::vector<std::vector<double>>((size_t)nvarh,
                                          std::vector<double>((size_t)max_ng)));

    std::vector<OldCellRec> result;
    result.reserve(65536);

    for (int32_t ilevel = 1; ilevel <= levelmax; ++ilevel) {
        double dx = std::pow(0.5, (double)ilevel);
        double xc[8][3] = {};
        for (int ind = 0; ind < twotondim; ++ind) {
            int iz = ind / 4;
            int iy = (ind - 4*iz) / 2;
            int ix = ind - 2*iy - 4*iz;
            xc[ind][0] = ((double)ix - 0.5) * dx;
            xc[ind][1] = ((double)iy - 0.5) * dx;
            xc[ind][2] = (ndim >= 3) ? ((double)iz - 0.5) * dx : 0.0;
        }

        for (int32_t j = 0; j < ndomain; ++j) {
            int32_t ngrida = get_ng(j, ilevel - 1);
            bool    own    = (j == icpu - 1);

            if (ngrida > 0) {
                frec_skip(fa); frec_skip(fa); frec_skip(fa);
                if (own) {
                    for (int d = 0; d < ndim && d < 3; ++d) {
                        frec_read(fa, xg_tmp.data(), ngrida);
                        std::copy(xg_tmp.begin(), xg_tmp.begin()+ngrida,
                                  xg[d].begin());
                    }
                    if (ndim > 3) for (int d = 3; d < ndim; ++d) frec_skip(fa);
                } else {
                    for (int d = 0; d < ndim; ++d) frec_skip(fa);
                }
                frec_skip(fa);
                for (int k = 0; k < 2*ndim; ++k) frec_skip(fa);
                if (own) {
                    for (int ind = 0; ind < twotondim; ++ind)
                        frec_read(fa, son_arr[(size_t)ind].data(), ngrida);
                } else {
                    for (int ind = 0; ind < twotondim; ++ind) frec_skip(fa);
                }
                for (int k = 0; k < 2*twotondim; ++k) frec_skip(fa);
            }

            frec_skip(fh); frec_skip(fh);

            if (ngrida > 0) {
                if (own) {
                    for (int ind = 0; ind < twotondim; ++ind)
                        for (int iv = 0; iv < nvarh; ++iv)
                            frec_read(fh, hd_arr[(size_t)ind][(size_t)iv].data(),
                                      ngrida);

                    for (int32_t k = 0; k < ngrida; ++k) {
                        for (int ind = 0; ind < twotondim; ++ind) {
                            if (son_arr[(size_t)ind][(size_t)k] != 0) continue;
                            OldCellRec c;
                            c.x = xg[0][(size_t)k] + xc[ind][0];
                            c.y = (ndim >= 2) ? xg[1][(size_t)k]+xc[ind][1] : 0.0;
                            c.z = (ndim >= 3) ? xg[2][(size_t)k]+xc[ind][2] : 0.0;
                            c.dx    = dx;
                            c.level = ilevel;
                            c.nvar  = nvar_use;
                            for (int iv = 0; iv < nvar_use; ++iv)
                                c.var[iv] = hd_arr[(size_t)ind][(size_t)iv][(size_t)k];
                            result.push_back(c);
                        }
                    }
                } else {
                    for (int k = 0; k < twotondim*nvarh; ++k) frec_skip(fh);
                }
            }
        }
    }

    fclose(fa); fclose(fh);
    return result;
}

// ---------------------------------------------------------------------------
// CellPhys extraction
// ---------------------------------------------------------------------------
static CellPhys extract_cell_hdf(
    const CellRec &c,
    double xc, double yc, double zc,
    double vxc, double vyc, double vzc,
    const CosmoInfo &cosmo, double mu_mean)
{
    CellPhys p;
    const double kpc = cosmo.kpc_per_code;
    p.x_kpc = (c.x-xc)*kpc; p.y_kpc = (c.y-yc)*kpc; p.z_kpc = (c.z-zc)*kpc;
    p.d_kpc = std::sqrt(p.x_kpc*p.x_kpc + p.y_kpc*p.y_kpc + p.z_kpc*p.z_kpc);

    double dx_k = kpc / (double)(int64_t(1) << c.level);
    p.vol_kpc3  = dx_k*dx_k*dx_k;

    const double v2u  = cosmo.unit_l*cosmo.unit_l/cosmo.unit_t/cosmo.unit_t;
    const double vkms = std::sqrt(v2u)*1e-5;

    double rho_c = (c.rho > 0.0f) ? (double)c.rho : 1e-100;
    p.vx_kms = (double)c.vx*vkms - vxc;
    p.vy_kms = (double)c.vy*vkms - vyc;
    p.vz_kms = (double)c.vz*vkms - vzc;
    double Prho = (rho_c > 0.0) ? std::max((double)c.pres/rho_c, 0.0) : 0.0;

    p.rho_cgs   = rho_c * cosmo.unit_d;
    p.nH_cc     = p.rho_cgs * X_H / mH_cgs;
    p.mass_msun = rho_c * (dx_k/kpc)*(dx_k/kpc)*(dx_k/kpc) * cosmo.Msun_per_code;
    p.metal     = (double)c.metal;
    p.T_K       = Prho * v2u * mu_mean * mH_cgs / kB_cgs;
    if (p.T_K < 10.0) p.T_K = 10.0;
    p.KE_kms2   = 0.5*(p.vx_kms*p.vx_kms + p.vy_kms*p.vy_kms + p.vz_kms*p.vz_kms);
    p.UE_kms2   = Prho/(gamma_gas-1.0)*v2u*1e-10;
    p.PE_kms2   = (double)c.pot * v2u*1e-10;
    p.vdot      = p.x_kpc*p.vx_kms + p.y_kpc*p.vy_kms + p.z_kpc*p.vz_kms;

    double h    = cosmo.H0/100.0;
    double den2 = p.nH_cc*mH_cgs/Msun_cgs*std::pow(kpc_cm,3.0)/(h*h)/1e10;
    if (den2 < 1e-100) den2 = 1e-100;
    p.cold      = (p.T_K < std::pow(10.0, 6.0+0.25*std::log10(den2)))
               || (p.nH_cc > 10.0);
    p.celltype  = -1;
    return p;
}

static CellPhys extract_cell_old(
    const OldCellRec &c,
    double xc, double yc, double zc,
    double vxc, double vyc, double vzc,
    const CosmoInfo &cosmo, double mu_mean)
{
    CellPhys p;
    const double kpc = cosmo.kpc_per_code;
    p.x_kpc = (c.x-xc)*kpc; p.y_kpc = (c.y-yc)*kpc; p.z_kpc = (c.z-zc)*kpc;
    p.d_kpc = std::sqrt(p.x_kpc*p.x_kpc + p.y_kpc*p.y_kpc + p.z_kpc*p.z_kpc);

    double dx_k = c.dx * kpc;
    p.vol_kpc3  = dx_k*dx_k*dx_k;

    const double v2u  = cosmo.unit_l*cosmo.unit_l/cosmo.unit_t/cosmo.unit_t;
    const double vkms = std::sqrt(v2u)*1e-5;

    double rho_c = (c.nvar > 0 && c.var[0] > 0.0) ? c.var[0] : 1e-100;
    double vx_c  = (c.nvar > 1) ? c.var[1]/rho_c : 0.0;
    double vy_c  = (c.nvar > 2) ? c.var[2]/rho_c : 0.0;
    double vz_c  = (c.nvar > 3) ? c.var[3]/rho_c : 0.0;
    double e_tot = (c.nvar > 4) ? c.var[4]/rho_c : 0.0;
    double e_kin = 0.5*(vx_c*vx_c + vy_c*vy_c + vz_c*vz_c);
    double Prho  = (gamma_gas-1.0)*std::max(e_tot - e_kin, 0.0);

    p.rho_cgs   = rho_c * cosmo.unit_d;
    p.nH_cc     = p.rho_cgs * X_H / mH_cgs;
    p.mass_msun = rho_c * c.dx*c.dx*c.dx * cosmo.Msun_per_code;
    p.metal     = (c.nvar > 5) ? c.var[5]/rho_c : 0.0;
    p.vx_kms    = vx_c*vkms - vxc;
    p.vy_kms    = vy_c*vkms - vyc;
    p.vz_kms    = vz_c*vkms - vzc;
    p.T_K       = Prho * v2u * mu_mean * mH_cgs / kB_cgs;
    if (p.T_K < 10.0) p.T_K = 10.0;
    p.KE_kms2   = 0.5*(p.vx_kms*p.vx_kms + p.vy_kms*p.vy_kms + p.vz_kms*p.vz_kms);
    p.UE_kms2   = Prho/(gamma_gas-1.0)*v2u*1e-10;
    p.PE_kms2   = 0.0;  // no stored potential in old Ramses format
    p.vdot      = p.x_kpc*p.vx_kms + p.y_kpc*p.vy_kms + p.z_kpc*p.vz_kms;

    double h    = cosmo.H0/100.0;
    double den2 = p.nH_cc*mH_cgs/Msun_cgs*std::pow(kpc_cm,3.0)/(h*h)/1e10;
    if (den2 < 1e-100) den2 = 1e-100;
    p.cold      = (p.T_K < std::pow(10.0, 6.0+0.25*std::log10(den2)))
               || (p.nH_cc > 10.0);
    p.celltype  = -1;
    return p;
}

// ---------------------------------------------------------------------------
// Potential reference + ISM/CGM/ICM classification.
// phi_ref = volume-weighted mean potential at 0.8*grav_r < d <= grav_r shell.
// ---------------------------------------------------------------------------
static void apply_pot_ref_and_classify(
    std::vector<CellPhys> &cells, double grav_r_kpc)
{
    if (cells.empty()) return;

    double sum_phi = 0.0, sum_vol = 0.0;
    for (auto &c : cells) {
        if (c.d_kpc >= 0.8*grav_r_kpc && c.d_kpc <= grav_r_kpc) {
            sum_phi += c.PE_kms2 * c.vol_kpc3;
            sum_vol += c.vol_kpc3;
        }
    }
    double phi_ref = 0.0;
    if (sum_vol > 0.0) {
        phi_ref = sum_phi / sum_vol;
    } else {
        // Fallback: use PE of the outermost cell
        double dmax = -1.0;
        for (auto &c : cells)
            if (c.d_kpc > dmax) { dmax = c.d_kpc; phi_ref = c.PE_kms2; }
    }
    for (auto &c : cells) c.PE_kms2 -= phi_ref;

    // ISM: total energy < 0
    for (auto &c : cells)
        if (c.PE_kms2 + c.KE_kms2 + c.UE_kms2 < 0.0) c.celltype = 1;

    // CGM/ICM distinction via ISM metallicity radial profile
    double d_shell = (grav_r_kpc > 0.0) ? grav_r_kpc / N_SHELL : 1.0;
    struct SM { double sum_Zm=0, sum_m=0; };
    std::vector<SM> sm(N_SHELL);
    for (auto &c : cells) {
        if (c.celltype != 1) continue;
        int s = std::min((int)(c.d_kpc/d_shell), N_SHELL-1);
        sm[(size_t)s].sum_Zm += c.metal*c.mass_msun;
        sm[(size_t)s].sum_m  += c.mass_msun;
    }
    std::vector<double> Z_mean(N_SHELL, 1e8), Z_std(N_SHELL, 0.0);
    for (int s = 0; s < N_SHELL; ++s)
        if (sm[(size_t)s].sum_m > 0.0)
            Z_mean[(size_t)s] = sm[(size_t)s].sum_Zm / sm[(size_t)s].sum_m;
    for (auto &c : cells) {
        if (c.celltype != 1) continue;
        int s = std::min((int)(c.d_kpc/d_shell), N_SHELL-1);
        if (Z_mean[(size_t)s] < 1e7) {
            double dz = c.metal - Z_mean[(size_t)s];
            Z_std[(size_t)s] += dz*dz * c.mass_msun / sm[(size_t)s].sum_m;
        }
    }
    for (int s = 0; s < N_SHELL; ++s)
        Z_std[(size_t)s] = std::sqrt(std::max(Z_std[(size_t)s], 0.0));
    for (int s = 1; s < N_SHELL; ++s)
        if (Z_mean[(size_t)s] >= 1e7) Z_mean[(size_t)s] = Z_mean[(size_t)s-1];

    for (auto &c : cells) {
        if (c.celltype == 1) continue;
        int s = std::min((int)(c.d_kpc/d_shell), N_SHELL-1);
        double Z_ism = (Z_mean[(size_t)s] < 1e7) ? Z_mean[(size_t)s] : minZval;
        double lowZ  = std::max(minZval, Z_ism - Z_std[(size_t)s]);
        c.celltype   = (c.metal > lowZ && (c.vdot > 0.0 || c.T_K > 1e10)) ? 0 : -1;
    }
}

static void accumulate_gasmass(
    const std::vector<CellPhys> &cells,
    int64_t gi, int32_t n_ap, const std::vector<double> &gas_r_kpc,
    GasPropResult &res)
{
    for (int32_t ai = 0; ai < n_ap; ++ai) {
        double r = gas_r_kpc[(size_t)ai];
        double mt=0,mh=0,mc=0, im=0,ih=0,ic=0, gm=0,gh=0,gc=0, xm=0,xh=0,xc2=0;
        for (auto &p : cells) {
            if (p.d_kpc > r) continue;
            double m = p.mass_msun;
            mt += m; if (p.cold) mc += m; else mh += m;
            if      (p.celltype==1){ im+=m; if(p.cold)ic+=m; else ih+=m; }
            else if (p.celltype==0){ gm+=m; if(p.cold)gc+=m; else gh+=m; }
            else                   { xm+=m; if(p.cold)xc2+=m; else xh+=m; }
        }
        size_t b = (size_t)(gi*n_ap+ai);
        res.gm_tot_tot[b]=mt; res.gm_tot_hot[b]=mh; res.gm_tot_cold[b]=mc;
        res.gm_ism_tot[b]=im; res.gm_ism_hot[b]=ih; res.gm_ism_cold[b]=ic;
        res.gm_cgm_tot[b]=gm; res.gm_cgm_hot[b]=gh; res.gm_cgm_cold[b]=gc;
        res.gm_icm_tot[b]=xm; res.gm_icm_hot[b]=xh; res.gm_icm_cold[b]=xc2;
    }
}

static void compute_pressure(
    const std::vector<CellPhys> &cells, double grav_r_kpc,
    double &ram_out, double &grav_out)
{
    double sum_rp=0, vol_icm=0, sum_gp=0, vol_ism=0;
    for (auto &p : cells) {
        if (p.d_kpc > grav_r_kpc) continue;
        double vol = p.vol_kpc3;
        if (p.celltype == -1) {
            double vv2 = (p.vx_kms*p.vx_kms+p.vy_kms*p.vy_kms+p.vz_kms*p.vz_kms)*1e10;
            sum_rp  += p.rho_cgs * vv2 * vol;
            vol_icm += vol;
        }
        if (p.celltype == 1) {
            sum_gp  += p.rho_cgs * std::fabs(p.PE_kms2)*1e10 * vol;
            vol_ism += vol;
        }
    }
    ram_out  = (vol_icm > 0.0) ?        sum_rp / vol_icm / kB_cgs : 0.0;
    grav_out = (vol_ism > 0.0) ? 0.5 * sum_gp / vol_ism / kB_cgs : 0.0;
}

// Finalize one galaxy: classify, compute pressures, accumulate gas masses.
// Then clear its cell data to free memory.
static void finalize_galaxy(
    int64_t gi,
    std::vector<CellPhys> &cells,
    const std::vector<double> &gas_r_kpc,
    double grav_r_kpc, int32_t n_gas_ap, bool need_grav,
    GasPropResult &res)
{
    if (cells.empty()) return;

    if (need_grav && grav_r_kpc > 0.0) {
        std::vector<CellPhys> inner, outer;
        inner.reserve(cells.size());
        outer.reserve(cells.size() / 4);
        for (auto &c : cells) {
            if (c.d_kpc <= grav_r_kpc) inner.push_back(c);
            else                        outer.push_back(c);
        }
        apply_pot_ref_and_classify(inner, grav_r_kpc);
        compute_pressure(inner, grav_r_kpc,
                         res.ram_pressure[(size_t)gi],
                         res.grav_pressure[(size_t)gi]);
        cells = std::move(inner);
        for (auto &c : outer) { c.celltype = -1; cells.push_back(c); }
    }
    accumulate_gasmass(cells, gi, n_gas_ap, gas_r_kpc, res);

    // Free cell memory immediately
    cells.clear();
    cells.shrink_to_fit();
}

} // anonymous namespace

// ===========================================================================
// compute_gasprop
// ===========================================================================
GasPropResult compute_gasprop(
    const vpp_set::Settings &vh,
    int32_t                   snap,
    const RawCatalog         &cat,
    const CosmoInfo          &cosmo)
{
    GasPropResult res;
    const int64_t n_gal    = cat.n;
    const int32_t n_gas_ap = (int32_t)vh.gas_r.size();
    res.n_gal = n_gal; res.n_gas_ap = n_gas_ap;
    if (n_gal == 0 || n_gas_ap == 0) return res;

    auto alloc = [&](std::vector<double> &v){ v.assign((size_t)(n_gal*n_gas_ap), 0.0); };
    alloc(res.gm_tot_tot); alloc(res.gm_tot_hot); alloc(res.gm_tot_cold);
    alloc(res.gm_ism_tot); alloc(res.gm_ism_hot); alloc(res.gm_ism_cold);
    alloc(res.gm_cgm_tot); alloc(res.gm_cgm_hot); alloc(res.gm_cgm_cold);
    alloc(res.gm_icm_tot); alloc(res.gm_icm_hot); alloc(res.gm_icm_cold);
    res.ram_pressure .assign((size_t)n_gal, 0.0);
    res.grav_pressure.assign((size_t)n_gal, 0.0);

    const std::string rref_col = (vh.horg == 'g') ? "R_HalfMass" : "R_200crit";
    auto chk = [&](const char *n){ return cat.f64.count(n) > 0; };
    if (!chk("Xc")||!chk("Yc")||!chk("Zc")||!chk(rref_col.c_str())) {
        LOG() << "[gasprop] ERROR: missing Xc/Yc/Zc or " << rref_col;
        return res;
    }
    bool has_vel = chk("VXc") && chk("VYc") && chk("VZc");
    const auto &cat_xc   = cat.f64.at("Xc");
    const auto &cat_yc   = cat.f64.at("Yc");
    const auto &cat_zc   = cat.f64.at("Zc");
    const auto &cat_rref = cat.f64.at(rref_col);
    const double inv_kpc = 1.0 / cosmo.kpc_per_code;
    const bool need_grav = (vh.grav_r > 0.0);

    // Per-galaxy geometry (shared between both paths)
    struct GalInfo {
        double xc, yc, zc, vxc, vyc, vzc;
        double max_r_code, grav_r_kpc;
        std::vector<double> gas_r_kpc;
    };
    std::vector<GalInfo> ginfo((size_t)n_gal);
    for (int64_t gi = 0; gi < n_gal; ++gi) {
        GalInfo &g = ginfo[(size_t)gi];
        g.xc  = cat_xc [(size_t)gi]*inv_kpc;
        g.yc  = cat_yc [(size_t)gi]*inv_kpc;
        g.zc  = cat_zc [(size_t)gi]*inv_kpc;
        g.vxc = has_vel ? cat.f64.at("VXc")[(size_t)gi] : 0.0;
        g.vyc = has_vel ? cat.f64.at("VYc")[(size_t)gi] : 0.0;
        g.vzc = has_vel ? cat.f64.at("VZc")[(size_t)gi] : 0.0;
        double rref_code = cat_rref[(size_t)gi]*inv_kpc;
        double rref_kpc  = cat_rref[(size_t)gi];
        g.grav_r_kpc = need_grav ? vh.grav_r*rref_kpc : 0.0;
        double mr = need_grav ? vh.grav_r*rref_code : 0.0;
        g.gas_r_kpc.resize((size_t)n_gas_ap);
        for (int32_t ai = 0; ai < n_gas_ap; ++ai) {
            g.gas_r_kpc[(size_t)ai] = vh.gas_r[(size_t)ai]*rref_kpc;
            mr = std::max(mr, vh.gas_r[(size_t)ai]*rref_code);
        }
        g.max_r_code = mr;
    }

    // ===================================================================
    // HDF5 streaming path (newramses=1)
    // ===================================================================
#ifdef VPP_USE_HDF5
    if (vh.newramses != 0) {
        std::ostringstream oss;
        oss << vh.dir_raw << "/cell_"
            << std::setw(5) << std::setfill('0') << snap << ".h5";
        const std::string cell_fn = oss.str();

        hid_t fid = H5Fopen(cell_fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid < 0) { LOG()<<"[gasprop] ERROR: cannot open "<<cell_fn; return res; }

        struct HBEntry { uint64_t hi, lo; };
        std::vector<HBEntry> hb_raw(8001);
        {
            hid_t ds = H5Dopen(fid, "leaf/hilbert_boundary", H5P_DEFAULT);
            if (ds < 0) { LOG()<<"[gasprop] ERROR: leaf/hilbert_boundary"; H5Fclose(fid); return res; }
            hid_t mt = H5Tcreate(H5T_COMPOUND, sizeof(HBEntry));
            H5Tinsert(mt, "hi", offsetof(HBEntry,hi), H5T_NATIVE_UINT64);
            H5Tinsert(mt, "lo", offsetof(HBEntry,lo), H5T_NATIVE_UINT64);
            H5Dread(ds, mt, H5S_ALL, H5S_ALL, H5P_DEFAULT, hb_raw.data());
            H5Tclose(mt); H5Dclose(ds);
        }
        std::vector<uint64_t> hb_lo(8001);
        for (int i = 0; i < 8001; ++i) hb_lo[i] = hb_raw[i].lo;

        std::vector<int32_t> cb(8001);
        {
            hid_t ds = H5Dopen(fid, "leaf/chunk_boundary", H5P_DEFAULT);
            if (ds < 0) { LOG()<<"[gasprop] ERROR: leaf/chunk_boundary"; H5Fclose(fid); return res; }
            H5Dread(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, cb.data());
            H5Dclose(ds);
        }

        int32_t levmax = 0;
        { uint64_t v = hb_lo[8000]; while (v > 1) { v >>= 3; ++levmax; } }
        LOG() << "[gasprop] " << cell_fn
              << "  levmax=" << levmax << "  total_leaf=" << cb[8000];

        // Pre-compute needed chunks per galaxy
        std::vector<std::vector<int32_t>> gal_chunks((size_t)n_gal);
        for (int64_t gi = 0; gi < n_gal; ++gi) {
            if (ginfo[(size_t)gi].max_r_code <= 0.0) continue;
            const GalInfo &g = ginfo[(size_t)gi];
            gal_chunks[(size_t)gi] = find_needed_chunks(
                g.xc, g.yc, g.zc, g.max_r_code, hb_lo, levmax);
        }

        // chunk -> galaxy list
        std::vector<std::vector<int32_t>> chunk_gals(8000);
        for (int64_t gi = 0; gi < n_gal; ++gi)
            for (int32_t ci : gal_chunks[(size_t)gi])
                chunk_gals[(size_t)ci].push_back((int32_t)gi);
        {
            int32_t nact = 0;
            for (auto &v : chunk_gals) if (!v.empty()) ++nact;
            LOG() << "[gasprop] n_active_chunks=" << nact;
        }

        // Remaining chunk counter for early finalization
        std::vector<int32_t> remaining((size_t)n_gal, 0);
        for (int64_t gi = 0; gi < n_gal; ++gi)
            remaining[(size_t)gi] = (int32_t)gal_chunks[(size_t)gi].size();

        // Per-galaxy cell accumulators
        std::vector<std::vector<CellPhys>> gal_cells((size_t)n_gal);
        for (int64_t gi = 0; gi < n_gal; ++gi)
            if (!gal_chunks[(size_t)gi].empty())
                gal_cells[(size_t)gi].reserve(8192);

        hid_t data_ds = H5Dopen(fid, "leaf/data", H5P_DEFAULT);
        if (data_ds < 0) { LOG()<<"[gasprop] ERROR: leaf/data missing"; H5Fclose(fid); return res; }

        hid_t cell_mt = H5Tcreate(H5T_COMPOUND, sizeof(CellRec));
        H5Tinsert(cell_mt, "position_x", offsetof(CellRec,x),     H5T_NATIVE_DOUBLE);
        H5Tinsert(cell_mt, "position_y", offsetof(CellRec,y),     H5T_NATIVE_DOUBLE);
        H5Tinsert(cell_mt, "position_z", offsetof(CellRec,z),     H5T_NATIVE_DOUBLE);
        H5Tinsert(cell_mt, "level",      offsetof(CellRec,level), H5T_NATIVE_INT32);
        H5Tinsert(cell_mt, "density",    offsetof(CellRec,rho),   H5T_NATIVE_FLOAT);
        H5Tinsert(cell_mt, "velocity_x", offsetof(CellRec,vx),    H5T_NATIVE_FLOAT);
        H5Tinsert(cell_mt, "velocity_y", offsetof(CellRec,vy),    H5T_NATIVE_FLOAT);
        H5Tinsert(cell_mt, "velocity_z", offsetof(CellRec,vz),    H5T_NATIVE_FLOAT);
        H5Tinsert(cell_mt, "pressure",   offsetof(CellRec,pres),  H5T_NATIVE_FLOAT);
        H5Tinsert(cell_mt, "metallicity",offsetof(CellRec,metal), H5T_NATIVE_FLOAT);
        H5Tinsert(cell_mt, "potential",  offsetof(CellRec,pot),   H5T_NATIVE_FLOAT);

        hid_t fspace = H5Dget_space(data_ds);

        std::vector<CellRec> chunk_buf;
        for (int32_t ci = 0; ci < 8000; ++ci) {
            const std::vector<int32_t> &gals = chunk_gals[(size_t)ci];
            if (gals.empty()) continue;

            hsize_t c_start = (hsize_t)cb[(size_t)ci];
            hsize_t c_count = (hsize_t)(cb[(size_t)ci+1] - cb[(size_t)ci]);
            if (c_count == 0) continue;

            chunk_buf.resize((size_t)c_count);
            hid_t mspace = H5Screate_simple(1, &c_count, nullptr);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET,
                                &c_start, nullptr, &c_count, nullptr);
            H5Dread(data_ds, cell_mt, mspace, fspace, H5P_DEFAULT, chunk_buf.data());
            H5Sclose(mspace);

#ifdef VPP_USE_OMP
#pragma omp parallel for schedule(dynamic,1)
#endif
            for (int32_t g_idx = 0; g_idx < (int32_t)gals.size(); ++g_idx) {
                int32_t gi = gals[(size_t)g_idx];
                const GalInfo &g = ginfo[(size_t)gi];
                const double r2  = g.max_r_code * g.max_r_code;

                std::vector<CellPhys> local;
                for (hsize_t k = 0; k < c_count; ++k) {
                    const CellRec &c = chunk_buf[k];
                    double dx = c.x-g.xc, dy = c.y-g.yc, dz = c.z-g.zc;
                    if (dx*dx+dy*dy+dz*dz > r2) continue;
                    local.push_back(extract_cell_hdf(
                        c, g.xc, g.yc, g.zc,
                        g.vxc, g.vyc, g.vzc, cosmo, vh.mu_mean));
                }
                auto &dest = gal_cells[(size_t)gi];
                dest.insert(dest.end(), local.begin(), local.end());
            }
            // OMP barrier is implicit at end of parallel for

            // Early finalization: finalize any galaxy whose last chunk was just read
            for (int32_t gi : gals) {
                if (--remaining[(size_t)gi] == 0) {
                    finalize_galaxy(gi, gal_cells[(size_t)gi],
                                    ginfo[(size_t)gi].gas_r_kpc,
                                    ginfo[(size_t)gi].grav_r_kpc,
                                    n_gas_ap, need_grav, res);
                }
            }
        }

        H5Sclose(fspace);
        H5Tclose(cell_mt);
        H5Dclose(data_ds);
        H5Fclose(fid);

        // Safety pass for any galaxy not yet finalized (remaining_chunks was 0 initially)
        for (int64_t gi = 0; gi < n_gal; ++gi)
            if (!gal_cells[(size_t)gi].empty())
                finalize_galaxy(gi, gal_cells[(size_t)gi],
                                ginfo[(size_t)gi].gas_r_kpc,
                                ginfo[(size_t)gi].grav_r_kpc,
                                n_gas_ap, need_grav, res);

        LOG() << "[gasprop] HDF5 streaming done";
        return res;
    }
#endif // VPP_USE_HDF5

    // ===================================================================
    // Old Ramses streaming path (newramses=0)
    // Uses Hilbert-key domain lookup from info_XXXXX.txt.
    // No stored gravitational potential; PE=0, no ISM/CGM classification.
    // ===================================================================
    LOG() << "[gasprop] newramses=0: streaming from per-CPU binary files";

    auto snap_dir = [&]() {
        std::ostringstream o;
        o << vh.dir_raw << "/output_" << std::setw(5) << std::setfill('0') << snap;
        return o.str();
    };
    auto amr_name = [&](int32_t cpu) {
        std::ostringstream o;
        o << snap_dir() << "/amr_"   << std::setw(5) << std::setfill('0') << snap
          << ".out" << std::setw(5) << std::setfill('0') << cpu;
        return o.str();
    };
    auto hydro_name = [&](int32_t cpu) {
        std::ostringstream o;
        o << snap_dir() << "/hydro_" << std::setw(5) << std::setfill('0') << snap
          << ".out" << std::setw(5) << std::setfill('0') << cpu;
        return o.str();
    };

    CellParamsOld cp;
    if (!read_cell_params_old(snap_dir(), snap, cp)) {
        LOG() << "[gasprop] ERROR: cannot read old-format cell params";
        return res;
    }
    LOG() << "[gasprop] old-format: ncpu=" << cp.ncpu
          << "  levmax=" << cp.levelmax << "  nvarh=" << cp.nvarh;

    std::vector<double> hindex_lo, hindex_hi;
    if (!read_hindex_old(vh.dir_raw, snap, cp.ncpu, hindex_lo, hindex_hi)) {
        LOG() << "[gasprop] ERROR: cannot read hindex from info file";
        return res;
    }

    // Pre-compute needed CPU files per galaxy
    std::vector<std::vector<int32_t>> gal_cpus((size_t)n_gal);
    for (int64_t gi = 0; gi < n_gal; ++gi) {
        if (ginfo[(size_t)gi].max_r_code <= 0.0) continue;
        const GalInfo &g = ginfo[(size_t)gi];
        gal_cpus[(size_t)gi] = find_needed_cpus_old(
            g.xc, g.yc, g.zc, g.max_r_code,
            hindex_lo, hindex_hi, cp.levelmax);
    }

    // cpu (0-based) -> galaxy list
    std::vector<std::vector<int32_t>> cpu_gals((size_t)cp.ncpu);
    for (int64_t gi = 0; gi < n_gal; ++gi)
        for (int32_t ci : gal_cpus[(size_t)gi])
            cpu_gals[(size_t)ci].push_back((int32_t)gi);
    {
        int32_t nact = 0;
        for (auto &v : cpu_gals) if (!v.empty()) ++nact;
        LOG() << "[gasprop] old-format n_active_cpus=" << nact;
    }

    // Remaining CPU counter for early finalization
    std::vector<int32_t> remaining((size_t)n_gal, 0);
    for (int64_t gi = 0; gi < n_gal; ++gi)
        remaining[(size_t)gi] = (int32_t)gal_cpus[(size_t)gi].size();

    // Per-galaxy cell accumulators
    std::vector<std::vector<CellPhys>> gal_cells((size_t)n_gal);
    for (int64_t gi = 0; gi < n_gal; ++gi)
        if (!gal_cpus[(size_t)gi].empty())
            gal_cells[(size_t)gi].reserve(8192);

    // Loop over CPU files in order (1-based)
    for (int32_t ci = 0; ci < cp.ncpu; ++ci) {
        const std::vector<int32_t> &gals = cpu_gals[(size_t)ci];
        if (gals.empty()) continue;

        int32_t icpu = ci + 1;
        std::vector<OldCellRec> cpu_cells =
            read_cpu_cells_old(amr_name(icpu), hydro_name(icpu), icpu, cp);
        if (cpu_cells.empty()) {
            // Still need to decrement remaining so finalization fires correctly
            for (int32_t gi : gals)
                if (--remaining[(size_t)gi] == 0)
                    finalize_galaxy(gi, gal_cells[(size_t)gi],
                                    ginfo[(size_t)gi].gas_r_kpc,
                                    ginfo[(size_t)gi].grav_r_kpc,
                                    n_gas_ap, need_grav, res);
            continue;
        }

        const int64_t nc = (int64_t)cpu_cells.size();

#ifdef VPP_USE_OMP
#pragma omp parallel for schedule(dynamic,1)
#endif
        for (int32_t g_idx = 0; g_idx < (int32_t)gals.size(); ++g_idx) {
            int32_t gi = gals[(size_t)g_idx];
            const GalInfo &g = ginfo[(size_t)gi];
            const double r2  = g.max_r_code * g.max_r_code;

            std::vector<CellPhys> local;
            for (int64_t k = 0; k < nc; ++k) {
                const OldCellRec &c = cpu_cells[(size_t)k];
                double dx = c.x-g.xc, dy = c.y-g.yc, dz = c.z-g.zc;
                if (dx*dx+dy*dy+dz*dz > r2) continue;
                local.push_back(extract_cell_old(
                    c, g.xc, g.yc, g.zc,
                    g.vxc, g.vyc, g.vzc, cosmo, vh.mu_mean));
            }
            auto &dest = gal_cells[(size_t)gi];
            dest.insert(dest.end(), local.begin(), local.end());
        }
        // OMP barrier implicit

        for (int32_t gi : gals) {
            if (--remaining[(size_t)gi] == 0) {
                finalize_galaxy(gi, gal_cells[(size_t)gi],
                                ginfo[(size_t)gi].gas_r_kpc,
                                ginfo[(size_t)gi].grav_r_kpc,
                                n_gas_ap, need_grav, res);
            }
        }
    }

    // Safety pass
    for (int64_t gi = 0; gi < n_gal; ++gi)
        if (!gal_cells[(size_t)gi].empty())
            finalize_galaxy(gi, gal_cells[(size_t)gi],
                            ginfo[(size_t)gi].gas_r_kpc,
                            ginfo[(size_t)gi].grav_r_kpc,
                            n_gas_ap, need_grav, res);

    LOG() << "[gasprop] old-format streaming done";
    return res;
}

} // namespace Pipeline
