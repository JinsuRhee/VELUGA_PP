#pragma once
#include "global/allvar.h"
#include "pipeline/bprop.h"
#include "pipeline/cosmo.h"
#include "pipeline/gasprop.h"
#include "pipeline/pipeline.h"

namespace Pipeline {

// Write one snapshot's results to HDF5.
//
// Output path: <dir_catalog>/<Galaxy|Halo>/VR_<Galaxy|Halo>/snap_NNNN.hdf5
//   (sibling of the snap_NNNN/ directory that holds the VR properties files)
//
// File structure mirrors rv_save.pro / save_cat_conly.c:
//
//   Top-level datasets (header):
//     SFR_R, SFR_T, MAG_R, CONF_R, Flux_List
//     GAS_R  [n_gas_ap]   — gas aperture multiples (new)
//     GRAV_R [1]          — gravitational aperture (new)
//     Mass_tot, R_HalfMass, Mvir, Rvir, Mass_200crit, R_200crit  [n_gal]
//     ID     [n_gal]      int32
//     CONF_M, CONF_N      [n_gal × n_conf_ap]  (2-D)
//
//   Per-galaxy group  /ID_XXXXXX/
//     Aexp     scalar double
//     isclump  scalar int32  (-1 if not computed)
//
//   Per-galaxy galaxy properties  /ID_XXXXXX/G_Prop/
//     G_<col>             scalar per catalog column
//     G_SFR               [n_sfr]   ([-1.0] when not computed)
//     G_ABmag_<band>      [n_mag_ap] per flux band
//     G_SB_<band>         [n_mag_ap] per flux band
//     G_ConFrac_M         [n_conf_ap]
//     G_ConFrac_N         [n_conf_ap]
//     G_GasMass_tot_tot   [n_gas_ap]  ([-1.0] when gas_r empty / skip_gasprop)
//     G_GasMass_tot_hot   [n_gas_ap]
//     G_GasMass_tot_cold  [n_gas_ap]
//     G_GasMass_bound_tot [n_gas_ap]  (bound = ISM, Etot < 0)
//     G_GasMass_bound_hot [n_gas_ap]
//     G_GasMass_bound_cold[n_gas_ap]
//     G_GasMass_cgm_tot   [n_gas_ap]
//     G_GasMass_cgm_hot   [n_gas_ap]
//     G_GasMass_cgm_cold  [n_gas_ap]
//     G_GasMass_icm_tot   [n_gas_ap]
//     G_GasMass_icm_hot   [n_gas_ap]
//     G_GasMass_icm_cold  [n_gas_ap]
//     G_RamPressure       [1]  K/cm^3
//     G_GravPressure      [1]  K/cm^3
//
//   Per-galaxy particle properties  /ID_XXXXXX/P_Prop/
//     P_ID  [n_ptcl]  int64  (bound ++ unbound, sentinel-filtered)
//
// When HDF5 is not compiled in (VPP_USE_HDF5 not defined) the function is a no-op.
void save_result(
    const vpp_set::Settings &vh,
    int32_t                  snap,
    const RawCatalog        &cat,
    const PtclIDCatalog     &ptcl,
    const BPropResult       &bprop,
    const GasPropResult     &gasprop,
    const CosmoInfo         &cosmo);

} // namespace Pipeline
