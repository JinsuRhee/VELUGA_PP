#pragma once
#include "global/allvar.h"
#include "pipeline/rawdata.h"
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace Pipeline
{
    // Returns the snapshot directory (no trailing slash)
    inline std::string snap_dir(const vpp_set::Settings &vh, int32_t snap)
    {
        std::ostringstream oss;
        oss << vh.dir_catalog << "/"
            << (vh.horg == 'g' ? "Galaxy" : "Halo")
            << "/snap_" << std::setw(4) << std::setfill('0') << snap;
        return oss.str();
    }

    // Path to the first properties file — used for existence check
    inline std::string catalog_probe(const vpp_set::Settings &vh, int32_t snap)
    {
        return snap_dir(vh, snap) + "/"
             + (vh.horg == 'g' ? "galaxy" : "halo")
             + ".dat.properties.0";
    }

    //-----
    // Raw VR catalog (rv_RawCatalog)
    //-----
    struct RawCatalog {
        int64_t n = 0;
        std::unordered_map<std::string, std::vector<int64_t>> i64;
        std::unordered_map<std::string, std::vector<int32_t>> i32;
        std::unordered_map<std::string, std::vector<double>>  f64;
    };

    //-----
    // Particle ID catalog (rv_ReadID)
    //
    // p_id  : [id_bdn_0..n_bdn-1, id_ubd_0..n_ubd-1]   (concatenated)
    // b_ind : [n_obj * 2]  b_ind[j*2+0]=start, b_ind[j*2+1]=end (inclusive) into p_id
    // u_ind : [n_obj * 2]  same but offset into unbound section of p_id
    //         start > end means the object has no particles of that type
    //-----
    struct PtclIDCatalog {
        int64_t n_obj = 0;
        int64_t n_bdn = 0;
        int64_t n_ubd = 0;
        std::vector<int64_t> p_id;    // all IDs: bound then unbound
        std::vector<int64_t> b_ind;   // [n_obj * 2]: bound index pairs into p_id
        std::vector<int64_t> u_ind;   // [n_obj * 2]: unbound index pairs into p_id
    };

    RawCatalog    load_catalog (const vpp_set::Settings &vh, int32_t snap);
    PtclIDCatalog read_ptclid  (const vpp_set::Settings &vh, int32_t snap);
    void          member_match (const vpp_set::Settings &vh, int32_t snap);
    void          compute_bulk (const vpp_set::Settings &vh, int32_t snap,
                                const RawCatalog &cat, const PtclIDCatalog &ptcl);
    void          save         (const vpp_set::Settings &vh, int32_t snap);
}
