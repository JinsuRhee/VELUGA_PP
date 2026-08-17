#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace Pipeline {

// Balanced k-d tree over 3-D double-precision positions.
//
// Build: O(n log n), sequential.
// Query: read-only after build — safe to call from multiple OMP threads.
//
// Position arrays are NOT stored inside the tree.
// Pass them again at query time (same pointers used during build).
//
struct KDTree {
    struct Node {
        double  split_val;  // split coordinate
        int32_t axis;       // split axis (0/1/2); -1 = leaf
        int32_t left;       // child node indices (-1 = none)
        int32_t right;
        int32_t idx_lo;     // range in perm[] for leaf nodes
        int32_t idx_hi;
    };

    std::vector<Node>    nodes;
    std::vector<int32_t> perm;   // permuted original point indices

    // Build tree from SoA position arrays (n points).
    // leaf_size: max points per leaf node.
    void build(const double *x, const double *y, const double *z,
               int64_t n, int32_t leaf_size = 16);

    // Find all points with Euclidean distance ≤ r from (qx,qy,qz).
    // Appends original array indices into `out`.
    // Thread-safe (read-only on tree).
    void query_radius(const double *x, const double *y, const double *z,
                      double qx, double qy, double qz, double r,
                      std::vector<int32_t> &out) const;

    // Find k nearest neighbours.
    // Returns original indices sorted by distance² (nearest first).
    void query_knn(const double *x, const double *y, const double *z,
                   double qx, double qy, double qz, int32_t k,
                   std::vector<int32_t> &idx_out,
                   std::vector<double>  &d2_out) const;

    int64_t size()  const { return (int64_t)perm.size(); }
    bool    empty() const { return perm.empty(); }

private:
    int32_t build_node(const double *x, const double *y, const double *z,
                       int32_t lo, int32_t hi, int32_t leaf_size);

    void query_radius_rec(const double *x, const double *y, const double *z,
                          int32_t nid, double qx, double qy, double qz,
                          double r2, std::vector<int32_t> &out) const;

    using HeapEntry = std::pair<double, int32_t>;  // (dist², orig_idx)

    void query_knn_rec(const double *x, const double *y, const double *z,
                       int32_t nid, double qx, double qy, double qz,
                       int32_t k, std::vector<HeapEntry> &heap) const;
};

} // namespace Pipeline
