// cosmo.cxx — RAMSES cosmology utilities
// Reads info file; builds conformal-time and look-back-time tables.
// Conformal time convention (same as IDL veluga):
//   tau(a=1) = 0, tau(a<1) < 0
//   integrand: f(a) = 1 / (a^3 * sqrt(oM/a^3 + oL))
// LBT convention:
//   lbt(z=0) = 0, lbt increases with z
//   integrand: g(z) = 1 / ((1+z) * sqrt(oM*(1+z)^3 + oL))

#include "pipeline/cosmo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#ifdef VPP_USE_HDF5
#include <hdf5.h>
#endif

namespace Pipeline
{

    // ---------------------------------------------------------------------------
    // Utility
    // ---------------------------------------------------------------------------

    static double interp1(const std::vector<double> &xs,
                          const std::vector<double> &ys, double x)
    {
        if (x <= xs.front())
            return ys.front();
        if (x >= xs.back())
            return ys.back();
        auto it = std::lower_bound(xs.begin(), xs.end(), x);
        size_t hi = (size_t)(it - xs.begin());
        size_t lo = hi - 1;
        double t = (x - xs[lo]) / (xs[hi] - xs[lo]);
        return ys[lo] + t * (ys[hi] - ys[lo]);
    }

    // ---------------------------------------------------------------------------
    // read_info
    // ---------------------------------------------------------------------------

    CosmoInfo read_info(const std::string &dir_raw, int32_t snap)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "/output_%05d/info_%05d.txt", snap, snap);
        const std::string path = dir_raw + buf;

        std::ifstream in(path);
        if (!in)
        {
            LOG() << "[cosmo] ERROR: cannot open " << path;
            return {};
        }

        CosmoInfo c;
        std::string line;

        auto trim = [](const std::string &s) -> std::string
        {
            size_t a = s.find_first_not_of(" \t");
            size_t b = s.find_last_not_of(" \t\r\n");
            if (a == std::string::npos)
                return "";
            return s.substr(a, b - a + 1);
        };

        while (std::getline(in, line))
        {
            auto eq = line.find('=');
            if (eq == std::string::npos)
                continue;
            std::string key = trim(line.substr(0, eq));
            std::istringstream vss(line.substr(eq + 1));
            if (key == "unit_l")
                vss >> c.unit_l;
            else if (key == "unit_d")
                vss >> c.unit_d;
            else if (key == "unit_t")
                vss >> c.unit_t;
            else if (key == "aexp")
                vss >> c.aexp;
            else if (key == "H0")
                vss >> c.H0;
            else if (key == "omega_m")
                vss >> c.omega_m;
            else if (key == "omega_l")
                vss >> c.omega_l;
            else if (key == "omega_b")
                vss >> c.omega_b;
            else if (key == "boxlen")
                vss >> c.boxlen;
        }

        c.unit_m = c.unit_d * c.unit_l * c.unit_l * c.unit_l;
        c.kpc_per_code = c.unit_l / 3.086e21;  // 1 kpc = 3.086e21 cm
        c.Msun_per_code = c.unit_m / 1.989e33; // 1 Msun = 1.989e33 g

        LOG() << "[cosmo] snap=" << snap
              << "  aexp=" << c.aexp << "  H0=" << c.H0
              << "  oM=" << c.omega_m << "  oL=" << c.omega_l
              << "  kpc/code=" << c.kpc_per_code
              << "  Msun/code=" << c.Msun_per_code;
        return c;
    }

    // ---------------------------------------------------------------------------
    // build_cosm_table
    // ---------------------------------------------------------------------------

    CosmTable build_cosm_table(double oM, double oL, double H0)
    {
        constexpr int N = 10000;
        CosmTable tbl;
        tbl.sfact.resize(N);
        tbl.conft.resize(N);
        tbl.redsh.resize(N);
        tbl.lbt.resize(N);

        // Scale factors: 0.02 → 1.0 (ascending)
        for (int i = 0; i < N; ++i)
            tbl.sfact[i] = 0.02 + 0.98 * i / (N - 1);

        // Conformal time integrand: f(a) = 1/(a^3 * sqrt(oM/a^3 + oL))
        auto f_conf = [&](double a) -> double
        {
            return 1.0 / (a * a * a * std::sqrt(oM / (a * a * a) + oL));
        };

        // Build conft via trapezoid, starting from conft[N-1]=0 at a=1, going left
        tbl.conft[N - 1] = 0.0;
        for (int i = N - 2; i >= 0; --i)
        {
            double da = tbl.sfact[i + 1] - tbl.sfact[i]; // > 0
            // conft decreases as a decreases (more negative for smaller a)
            tbl.conft[i] = tbl.conft[i + 1] - da * 0.5 * (f_conf(tbl.sfact[i]) + f_conf(tbl.sfact[i + 1]));
        }
        // conft is now sorted ascending (most negative at [0], 0 at [N-1])

        // Redshift table: z ascending from ~0 to ~49
        for (int i = 0; i < N; ++i)
            tbl.redsh[i] = 1.0 / tbl.sfact[N - 1 - i] - 1.0;

        // LBT integrand: g(z) = 1/((1+z)*sqrt(oM*(1+z)^3 + oL))
        auto f_lbt = [&](double z) -> double
        {
            double op1z = 1.0 + z;
            return 1.0 / (op1z * std::sqrt(oM * op1z * op1z * op1z + oL));
        };

        // Hubble time in Gyr: 1/H0 [s] = 3.08568025e19 km / (H0 km/s/Mpc) / (3.1536e16 s/Gyr)
        const double H0_inv_Gyr = 3.08568025e19 / (H0 * 3.1536e16);

        // Build LBT via trapezoid from z=0 upward
        tbl.lbt[0] = 0.0;
        for (int i = 1; i < N; ++i)
        {
            double dz = tbl.redsh[i] - tbl.redsh[i - 1];
            tbl.lbt[i] = tbl.lbt[i - 1] + dz * 0.5 * (f_lbt(tbl.redsh[i - 1]) + f_lbt(tbl.redsh[i])) * H0_inv_Gyr;
        }

        return tbl;
    }

    // ---------------------------------------------------------------------------
    // conf_to_age_gyr_batch
    // ---------------------------------------------------------------------------

    void conf_to_age_gyr_batch(const float *tconf, float *age_gyr, int64_t n,
                               const CosmTable &tbl, double z_snap)
    {
        const double lbt_snap = interp1(tbl.redsh, tbl.lbt, z_snap);

#ifdef VPP_USE_OMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < n; ++i)
        {
            // 1. conft_birth → sfact_birth (conft ascending → sfact ascending)
            double sfact_b = interp1(tbl.conft, tbl.sfact, (double)tconf[i]);
            sfact_b = std::max(sfact_b, tbl.sfact.front());
            sfact_b = std::min(sfact_b, tbl.sfact.back());

            // 2. sfact_birth → z_birth → lbt_birth
            double z_b = 1.0 / sfact_b - 1.0;
            double lbt_b = interp1(tbl.redsh, tbl.lbt, z_b);

            // 3. age = LBT(birth) - LBT(snap)  (positive for stars born before snap)
            age_gyr[i] = (float)std::max(0.0, lbt_b - lbt_snap);
        }
    }

    // ---------------------------------------------------------------------------
    // read_info_hdf — read cosmo params from HDF5 root attributes (newramses=1)
    // ---------------------------------------------------------------------------

#ifdef VPP_USE_HDF5
    static double h5_attr_double(hid_t fid, const char *name)
    {
        double v = 0.0;
        hid_t aid = H5Aopen(fid, name, H5P_DEFAULT);
        if (aid < 0)
        {
            LOG() << "[cosmo] WARNING: attr '" << name << "' not found";
            return v;
        }
        H5Aread(aid, H5T_NATIVE_DOUBLE, &v);
        H5Aclose(aid);
        return v;
    }
#endif

    CosmoInfo read_info_hdf(const std::string &dir_raw, int32_t snap)
    {
#ifndef VPP_USE_HDF5
        LOG() << "[cosmo] read_info_hdf: HDF5 not compiled in";
        (void)dir_raw;
        (void)snap;
        return {};
#else
        char buf[64];
        std::snprintf(buf, sizeof(buf), "/part_%05d.h5", snap);
        const std::string path = dir_raw + buf;

        hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid < 0)
        {
            LOG() << "[cosmo] ERROR: cannot open " << path;
            return {};
        }

        CosmoInfo c;
        c.unit_l = h5_attr_double(fid, "unit_l");
        c.unit_d = h5_attr_double(fid, "unit_d");
        c.unit_t = h5_attr_double(fid, "unit_t");
        c.aexp = h5_attr_double(fid, "aexp");
        c.H0 = h5_attr_double(fid, "H0");
        c.omega_m = h5_attr_double(fid, "omega_m");
        c.omega_l = h5_attr_double(fid, "omega_l");
        c.omega_b = h5_attr_double(fid, "omega_b");
        c.boxlen = h5_attr_double(fid, "boxlen");
        H5Fclose(fid);

        c.unit_m = c.unit_d * c.unit_l * c.unit_l * c.unit_l;
        c.kpc_per_code = c.unit_l / 3.086e21;
        c.Msun_per_code = c.unit_m / 1.989e33;

        LOG() << "[cosmo] snap=" << snap << " (HDF5)"
              << "  aexp=" << c.aexp << "  H0=" << c.H0
              << "  oM=" << c.omega_m << "  oL=" << c.omega_l
              << "  kpc/code=" << c.kpc_per_code
              << "  Msun/code=" << c.Msun_per_code;
        return c;
#endif
    }

} // namespace Pipeline
