#pragma once
#include "global/allvar.h"
#include "pipeline/cosmo.h"
#include "pipeline/miles.h"
#include "pipeline/pipeline.h"
#include <cstdint>
#include <vector>

namespace Pipeline {

// Per-galaxy bulk photometric properties
// abmag / sbf layout: [gal * n_mag_ap * n_flux + ap * n_flux + fi]
// sfr layout:         [gal * n_sfr + si]
// isclump layout:     [gal]  1=clump, -1=not a clump / not computed
//
// All arrays are always allocated to the correct size regardless of horg,
// so the output format is identical for horg='g' and horg='h'.
// For horg='h': abmag/sbf/sfr filled with NaN, isclump filled with -1.
struct BPropResult {
    int64_t n_gal     = 0;
    int32_t n_flux    = 0;
    int32_t n_mag_ap  = 0;
    int32_t n_sfr     = 0;
    int32_t n_conf_ap = 0;
    std::vector<double>  abmag;      // AB magnitude per galaxy × aperture × band
    std::vector<double>  sbf;        // surface brightness per galaxy × aperture × band [mag/pc^2]
    std::vector<double>  sfr;        // SFR [Msun/yr] per galaxy × SFR window
    std::vector<int32_t> isclump;    // 1=clump, -1=not a clump / not computed
    std::vector<double>  confrac_n;  // [gal * n_conf_ap + ai]: LR-DM number fraction in aperture
    std::vector<double>  confrac_m;  // [gal * n_conf_ap + ai]: LR-DM mass fraction in aperture
};

// Compute abmag, sbf, sfr for all galaxies (horg='g' only) and confrac (both horg).
//
// Particles are streamed per-chunk from disk (newramses=0: per-CPU Fortran binary files;
// newramses=1: HDF5 BATCH=4M chunks).  No pre-loaded RawData is required.
//
// Requires cat: Xc, Yc, Zc, R_HalfMass (and R_200crit for horg='h' confrac).
BPropResult compute_bprop(
    const vpp_set::Settings  &vh,
    int32_t                   snap,
    const RawCatalog         &cat,
    const PtclIDCatalog      &ptcl,
    const CosmoInfo          &cosmo,
    const MilesML            &miles,
    const CosmTable          &ctbl);

} // namespace Pipeline
