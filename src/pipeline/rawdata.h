#pragma once
#include "global/allvar.h"
#include "pipeline/kdtree.h"
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Pipeline {

// Star particle data (structure-of-arrays)
struct StarData {
    int64_t n = 0;
    std::vector<double>  x, y, z;
    std::vector<float>   vx, vy, vz;
    std::vector<float>   mass, age, metal;
    std::vector<int64_t> id;
};

// DM particle data (structure-of-arrays)
struct DMData {
    int64_t n = 0;
    std::vector<double>  x, y, z;
    std::vector<float>   vx, vy, vz;
    std::vector<float>   mass;
    std::vector<int64_t> id;
};

// Leaf-cell hydro data (structure-of-arrays, column-major hydro)
//
// Old Ramses binary: nvar hydro variables stored in RAMSES code units
//   var[0*n+i] = density  (rho)
//   var[1*n+i] = x-momentum (rho*vx)
//   var[2*n+i] = y-momentum (rho*vy)
//   var[3*n+i] = z-momentum (rho*vz)
//   var[4*n+i] = total energy (rho*T/mu in code units)
//   var[5*n+i] = first extra variable (typically metallicity*rho)
//   var[6+...]  = additional fields (dust, chemistry, ...)
//
// New Ramses HDF5: pre-processed fields (nvar=7)
//   var[0*n+i] = density
//   var[1*n+i] = velocity_x  (NOT momentum)
//   var[2*n+i] = velocity_y
//   var[3*n+i] = velocity_z
//   var[4*n+i] = pressure     (rho*T/mu, divide by density for T/mu)
//   var[5*n+i] = metallicity
//   var[6*n+i] = gravitational potential
struct CellData {
    int64_t n    = 0;
    int32_t nvar = 0;           // number of hydro variables per cell

    std::vector<double>  x, y, z;    // cell centre (box code units [0,1])
    std::vector<float>   dx;          // cell size   (box code units)
    std::vector<int32_t> level;       // AMR refinement level

    // Hydro: column-major [nvar * n]
    // Access cell i, variable v: var[v * n + i]
    std::vector<double> var;
};

// Full raw RAMSES snapshot
struct RawData {
    StarData star;
    DMData   dm;
    CellData cell;
    // Particle ID → index in star (horg='g') or dm (horg='h') array
    std::unordered_map<int64_t, int64_t> id_hash;
    // Spatial k-d trees — built by build_kdtrees() after read_raw()
    KDTree star_tree;
    KDTree dm_tree;
    KDTree cell_tree;
};

// Read all raw RAMSES data for a snapshot and build the ID hash table.
// Dispatches to old Fortran binary (newramses==0) or new HDF5 (newramses==1).
RawData read_raw(const vpp_set::Settings &vh, int32_t snap);

// Build k-d trees for star, dm, and cell positions.
// Call after read_raw(). Trees are read-only after build and safe for OMP queries.
// The three trees are built with OMP sections (up to 3-way parallelism).
void build_kdtrees(RawData &raw);

// Stream star+dm particles from old-format (newramses=0) per-CPU Fortran binary files.
// Each callback is called once per CPU file.  Return false from a callback to stop
// calling that callback for remaining files; star and DM can stop independently.
void stream_ptcl_old(
    const vpp_set::Settings &vh,
    int32_t snap,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const float *age, const float *metal,
                              const int64_t *id, int64_t n)> &star_cb,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const int64_t *id, int64_t n)> &dm_cb);

#ifdef VPP_USE_HDF5
// Stream star/dm particles from a newramses part HDF5 file in BATCH=4M chunks.
// cb is called once per batch; return false to stop streaming after the current batch.
// pointers are invalid after cb returns.
void stream_stars_hdf(
    const std::string &part_fn, int nthreads,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const float *age, const float *metal,
                              const int64_t *id, int64_t n)> &cb);

void stream_dm_hdf(
    const std::string &part_fn, int nthreads,
    const std::function<bool(const double *x, const double *y, const double *z,
                              const float *mass, const int64_t *id, int64_t n)> &cb);
#endif

} // namespace Pipeline
