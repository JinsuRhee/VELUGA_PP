// kdtree.cxx — balanced k-d tree for 3-D spatial queries
//
// Construction: O(n log n), sequential.
// Queries: read-only → thread-safe for simultaneous OMP queries.
//
// Algorithm:
//   Build:  recursive median split via std::nth_element (O(n) per level).
//           Split axis chosen as the dimension with the largest coordinate spread.
//   Query:  branch-and-bound traversal; prune a subtree when its closest
//           possible point is farther than the current best.

#include "pipeline/kdtree.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace Pipeline {

// ============================================================
// Build
// ============================================================

void KDTree::build(const double *x, const double *y, const double *z,
                   int64_t n, int32_t leaf_size)
{
    nodes.clear();
    perm.clear();
    if (n <= 0) return;

    perm.resize((size_t)n);
    std::iota(perm.begin(), perm.end(), 0);

    // Upper bound on node count: 2 * ceil(n / leaf_size) + 4
    nodes.reserve((size_t)(2 * (n / leaf_size + 1) + 4));

    build_node(x, y, z, 0, (int32_t)n, leaf_size);
}

int32_t KDTree::build_node(const double *x, const double *y, const double *z,
                            int32_t lo, int32_t hi, int32_t leaf_size)
{
    // Grab slot before recursion so the index stays stable
    const int32_t nid = (int32_t)nodes.size();
    nodes.push_back(Node{});
    nodes[nid].idx_lo = lo;
    nodes[nid].idx_hi = hi;
    nodes[nid].left   = -1;
    nodes[nid].right  = -1;

    if (hi - lo <= leaf_size) {
        nodes[nid].axis = -1;
        return nid;
    }

    // Find the axis with the largest coordinate spread
    double lo_c[3] = {+1e300, +1e300, +1e300};
    double hi_c[3] = {-1e300, -1e300, -1e300};
    for (int32_t i = lo; i < hi; ++i) {
        const int32_t p = perm[(size_t)i];
        lo_c[0] = std::min(lo_c[0], x[p]); hi_c[0] = std::max(hi_c[0], x[p]);
        lo_c[1] = std::min(lo_c[1], y[p]); hi_c[1] = std::max(hi_c[1], y[p]);
        lo_c[2] = std::min(lo_c[2], z[p]); hi_c[2] = std::max(hi_c[2], z[p]);
    }
    int axis = 0;
    if (hi_c[1] - lo_c[1] > hi_c[axis] - lo_c[axis]) axis = 1;
    if (hi_c[2] - lo_c[2] > hi_c[axis] - lo_c[axis]) axis = 2;

    // Partition at the median using nth_element (O(n) expected)
    const int32_t mid = (lo + hi) / 2;
    const double *crd = (axis == 0) ? x : (axis == 1) ? y : z;
    std::nth_element(perm.begin() + lo, perm.begin() + mid, perm.begin() + hi,
                     [crd](int32_t a, int32_t b){ return crd[a] < crd[b]; });

    nodes[nid].axis      = axis;
    nodes[nid].split_val = crd[perm[(size_t)mid]];

    const int32_t left_id  = build_node(x, y, z, lo,  mid, leaf_size);
    const int32_t right_id = build_node(x, y, z, mid, hi,  leaf_size);

    // nodes[] may have been reallocated by recursion; index by nid (still valid)
    nodes[(size_t)nid].left  = left_id;
    nodes[(size_t)nid].right = right_id;

    return nid;
}

// ============================================================
// query_radius
// ============================================================

void KDTree::query_radius(const double *x, const double *y, const double *z,
                           double qx, double qy, double qz, double r,
                           std::vector<int32_t> &out) const
{
    if (nodes.empty()) return;
    query_radius_rec(x, y, z, 0, qx, qy, qz, r * r, out);
}

void KDTree::query_radius_rec(const double *x, const double *y, const double *z,
                               int32_t nid, double qx, double qy, double qz,
                               double r2, std::vector<int32_t> &out) const
{
    const Node &nd = nodes[(size_t)nid];

    if (nd.axis < 0) {
        // Leaf: check every point
        for (int32_t i = nd.idx_lo; i < nd.idx_hi; ++i) {
            const int32_t p = perm[(size_t)i];
            const double dx = x[p] - qx, dy = y[p] - qy, dz = z[p] - qz;
            if (dx*dx + dy*dy + dz*dz <= r2)
                out.push_back(p);
        }
        return;
    }

    // Distance from query point to split hyperplane
    const double qcrd  = (nd.axis == 0) ? qx : (nd.axis == 1) ? qy : qz;
    const double dsplit = qcrd - nd.split_val;   // signed

    // Visit near side first, then prune far side if hyperplane is beyond radius
    const int32_t near = (dsplit <= 0.0) ? nd.left  : nd.right;
    const int32_t far  = (dsplit <= 0.0) ? nd.right : nd.left;

    query_radius_rec(x, y, z, near, qx, qy, qz, r2, out);
    if (dsplit * dsplit <= r2)
        query_radius_rec(x, y, z, far,  qx, qy, qz, r2, out);
}

// ============================================================
// query_knn
// ============================================================

void KDTree::query_knn(const double *x, const double *y, const double *z,
                        double qx, double qy, double qz, int32_t k,
                        std::vector<int32_t> &idx_out,
                        std::vector<double>  &d2_out) const
{
    idx_out.clear(); d2_out.clear();
    if (nodes.empty() || k <= 0) return;

    // Max-heap of (dist², orig_idx) — top is the current worst best
    std::vector<HeapEntry> heap;
    heap.reserve((size_t)k + 1);

    query_knn_rec(x, y, z, 0, qx, qy, qz, k, heap);

    // Sort nearest-first
    std::sort_heap(heap.begin(), heap.end());
    idx_out.reserve(heap.size()); d2_out.reserve(heap.size());
    for (auto &e : heap) {
        d2_out.push_back(e.first);
        idx_out.push_back(e.second);
    }
}

void KDTree::query_knn_rec(const double *x, const double *y, const double *z,
                             int32_t nid, double qx, double qy, double qz,
                             int32_t k, std::vector<HeapEntry> &heap) const
{
    const Node &nd = nodes[(size_t)nid];

    if (nd.axis < 0) {
        // Leaf: update max-heap
        for (int32_t i = nd.idx_lo; i < nd.idx_hi; ++i) {
            const int32_t p = perm[(size_t)i];
            const double dx = x[p] - qx, dy = y[p] - qy, dz = z[p] - qz;
            const double d2 = dx*dx + dy*dy + dz*dz;
            if ((int32_t)heap.size() < k) {
                heap.push_back({d2, p});
                std::push_heap(heap.begin(), heap.end());
            } else if (d2 < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {d2, p};
                std::push_heap(heap.begin(), heap.end());
            }
        }
        return;
    }

    const double qcrd  = (nd.axis == 0) ? qx : (nd.axis == 1) ? qy : qz;
    const double dsplit = qcrd - nd.split_val;

    const int32_t near = (dsplit <= 0.0) ? nd.left  : nd.right;
    const int32_t far  = (dsplit <= 0.0) ? nd.right : nd.left;

    query_knn_rec(x, y, z, near, qx, qy, qz, k, heap);

    // Prune far branch if the hyperplane is farther than our current worst best
    const double worst = ((int32_t)heap.size() < k) ? 1e300 : heap.front().first;
    if (dsplit * dsplit < worst)
        query_knn_rec(x, y, z, far, qx, qy, qz, k, heap);
}

} // namespace Pipeline
