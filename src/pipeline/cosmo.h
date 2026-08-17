#pragma once
#include "global/allvar.h"
#include <cstdint>
#include <string>
#include <vector>

namespace Pipeline {

// Unit-system and cosmological parameters from RAMSES info file
struct CosmoInfo {
    double unit_l   = 0.0;  // code length unit in cm
    double unit_d   = 0.0;  // code density unit in g/cm^3
    double unit_m   = 0.0;  // code mass unit in g (= unit_d * unit_l^3)
    double unit_t   = 0.0;  // code time unit in s
    double aexp     = 1.0;  // expansion factor at snapshot
    double H0       = 70.0; // km/s/Mpc
    double omega_m  = 0.3;
    double omega_l  = 0.7;
    double omega_b  = 0.0;  // baryon density parameter (from info file)
    double boxlen   = 1.0;  // box size in Mpc/h (RAMSES namelist value)
    double kpc_per_code  = 0.0;  // = unit_l / 3.0857e21
    double Msun_per_code = 0.0;  // = unit_m / 1.989e33
};

// Lookup tables for conformal time ↔ scale factor and redshift ↔ lookback time
struct CosmTable {
    // Conformal time table: sfact ascending [0.02,1.0], conft ascending [neg,0]
    std::vector<double> sfact;
    std::vector<double> conft;
    // LBT table: redsh ascending [0,~50], lbt ascending [0,~13.8] Gyr
    std::vector<double> redsh;
    std::vector<double> lbt;
};

// Read RAMSES info file for a given snapshot (old Fortran binary format)
CosmoInfo read_info    (const std::string &dir_raw, int32_t snap);
// Read cosmological parameters from HDF5 root attributes (newramses=1)
CosmoInfo read_info_hdf(const std::string &dir_raw, int32_t snap);

// Build conformal-time / LBT tables from cosmological parameters
CosmTable build_cosm_table(double oM, double oL, double H0);

// Convert RAMSES star birth conformal time to stellar age [Gyr] at the snapshot.
// tconf_birth < 0  (IDL convention: tau=0 at a=1, negative in the past)
// z_snap = 1/aexp - 1
void conf_to_age_gyr_batch(const float *tconf, float *age_gyr, int64_t n,
                            const CosmTable &tbl, double z_snap);

} // namespace Pipeline
