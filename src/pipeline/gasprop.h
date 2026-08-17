#pragma once
#include "global/allvar.h"
#include "pipeline/cosmo.h"
#include "pipeline/pipeline.h"
#include "pipeline/rawdata.h"
#include <cstdint>
#include <vector>

namespace Pipeline {

// Gas mass and pressure result per galaxy.
//
// Gas mass arrays: [n_gal * n_gas_ap + ai]
//   TYPE  : tot (all) | ism (bound, Etot<0) | cgm (unbound, high-Z outflowing)
//           | icm (unbound remainder = IGM)
//   PHASE : tot | hot | cold
//   Mass in Msun.
//
// Pressure scalars: [n_gal]
//   ram_pressure  : volume-weighted mean rho*v^2/kB  of ICM  cells in grav_r [K/cm^3]
//   grav_pressure : volume-weighted mean rho*|PE|/kB of ISM cells in grav_r [K/cm^3]
//   Both 0.0 when grav_r<=0 or no cells found.
//
// When skip_gasprop=1 or gas_r is empty, all arrays have size 0 and n_gas_ap=0.
// When grav_r<=0 potential is not computed; cells classified as unbound/bound only
// via total energy sign — in that case Etot=KE+UE (PE ignored).
struct GasPropResult {
    int64_t n_gal    = 0;
    int32_t n_gas_ap = 0;

    // per galaxy × per aperture
    std::vector<double> gm_tot_tot, gm_tot_hot, gm_tot_cold;
    std::vector<double> gm_ism_tot, gm_ism_hot, gm_ism_cold;
    std::vector<double> gm_cgm_tot, gm_cgm_hot, gm_cgm_cold;
    std::vector<double> gm_icm_tot, gm_icm_hot, gm_icm_cold;

    // per galaxy (grav_r aperture)
    std::vector<double> ram_pressure;
    std::vector<double> grav_pressure;
};

// Compute gas mass + pressure for all objects in the catalog.
// newramses=1: streams cells directly from cell_XXXXX.h5 using Hilbert-key chunk selection.
// newramses=0: falls back to reading all cells via read_raw/build_kdtrees internally.
// cat must contain: Xc, Yc, Zc, and R_HalfMass (horg='g') or R_200crit (horg='h').
// For relative velocity: VXc, VYc, VZc in km/s (optional; zeroed if missing).
GasPropResult compute_gasprop(
    const vpp_set::Settings &vh,
    int32_t                  snap,
    const RawCatalog        &cat,
    const CosmoInfo         &cosmo);

} // namespace Pipeline
