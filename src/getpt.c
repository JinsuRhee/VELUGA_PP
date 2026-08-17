/*
 * js_getpt_ft_conly.c  —  Pure C + OpenMP gravitational potential solver
 *
 * Entry: void js_getpt_ft(int argc, void *argv[])
 *   argv[0] = int    larr[20]
 *   argv[1] = double darr[20]
 *   argv[2] = double pos[n_ptcl * n_dim]   column-major: pos[k*n + i]
 *   argv[3] = double mm[n_ptcl]
 *   argv[4] = double pot[n_ptcl]   (output)
 *   argv[5] = double force[n_ptcl] (output)
 *
 *  larr[0]=n_ptcl  larr[2]=n_thread  larr[3]=p_type  larr[4]=e_type
 *  larr[12]=bsize  darr[0]=Gconst
 *
 * Performance design:
 *  - Tree build: 2-phase parallel (sequential coarse + OMP parallel subtrees)
 *  - Pot loops: branch-free inner loops + #pragma omp simd + SoA leaf data
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* column-major: dimension k, particle i (both 0-indexed) */
#define POS(i,k)  pos[(size_t)(k)*n_ptcl+(i)]

#define SWAPDBL(a,b) do{double _t=(a);(a)=(b);(b)=_t;}while(0)
#define SWAPINT(a,b) do{int   _t=(a);(a)=(b);(b)=_t;}while(0)

/* =========================================================
 *  Node (bstart/bend 0-indexed; left/right/parent 1-indexed)
 * ========================================================= */
typedef struct {
    int    id, bstart, bend, ncount;
    int    splitdim, splitind, leaf;
    double splitval;
    double bnd[3][2];
    double cen[3];
    double mass;
    int    numnode, level;
    int    parent, sibling, left, right;
} Node;

/* =========================================================
 *  nth_element: partial sort so arr[target] == sorted value.
 *  idx[] carries the same swaps.
 * ========================================================= */
static void nth_element(double * restrict arr, int * restrict idx,
                        int lo, int hi, int target)
{
    while (lo < hi) {
        int m = lo + (hi - lo) / 2;
        if (arr[m]  < arr[lo]) { SWAPDBL(arr[m],  arr[lo]);  SWAPINT(idx[m],  idx[lo]);  }
        if (arr[hi] < arr[lo]) { SWAPDBL(arr[hi], arr[lo]);  SWAPINT(idx[hi], idx[lo]);  }
        if (arr[m]  < arr[hi]) { SWAPDBL(arr[m],  arr[hi]);  SWAPINT(idx[m],  idx[hi]);  }
        double pivot = arr[hi];
        int i = lo - 1;
        for (int j = lo; j < hi; j++) {
            if (arr[j] <= pivot) {
                i++;
                SWAPDBL(arr[i], arr[j]);
                SWAPINT(idx[i], idx[j]);
            }
        }
        i++;
        SWAPDBL(arr[i], arr[hi]);
        SWAPINT(idx[i], idx[hi]);
        if (i == target) return;
        if (i  < target) lo = i + 1;
        else             hi = i - 1;
    }
}

/* =========================================================
 *  Scratch buffer pool — avoids per-node malloc overhead.
 *  All scratch buffers sized for n_ptcl.
 * ========================================================= */
typedef struct {
    double *pos;   /* 3*n_ptcl */
    double *mm;    /* n_ptcl   */
    int    *org;   /* n_ptcl   */
    double *key;   /* n_ptcl   */
    int    *perm;  /* n_ptcl   */
} Scratch;

static Scratch scratch_alloc(int n) {
    Scratch s;
    s.pos  = (double *)malloc(3L * n * sizeof(double));
    s.mm   = (double *)malloc(n * sizeof(double));
    s.org  = (int    *)malloc(n * sizeof(int));
    s.key  = (double *)malloc(n * sizeof(double));
    s.perm = (int    *)malloc(n * sizeof(int));
    return s;
}
static void scratch_free(Scratch *s) {
    free(s->pos); free(s->mm); free(s->org); free(s->key); free(s->perm);
}

/* =========================================================
 *  Reorder segment [bstart..bstart+n-1] using perm[0..n-1].
 *  Uses pre-allocated scratch buffers to avoid malloc.
 * ========================================================= */
static void reorder_seg(double * restrict pos, double * restrict mm,
                        int * restrict orgind,
                        const int * restrict perm,
                        int n, int bstart, int n_ptcl,
                        Scratch *sc)
{
    for (int i = 0; i < n; i++) {
        int src = bstart + perm[i];
        sc->pos[0*n+i] = pos[(size_t)0*n_ptcl + src];
        sc->pos[1*n+i] = pos[(size_t)1*n_ptcl + src];
        sc->pos[2*n+i] = pos[(size_t)2*n_ptcl + src];
        sc->mm[i]      = mm[src];
        sc->org[i]     = orgind[src];
    }
    for (int i = 0; i < n; i++) {
        int dst = bstart + i;
        pos[(size_t)0*n_ptcl + dst] = sc->pos[0*n+i];
        pos[(size_t)1*n_ptcl + dst] = sc->pos[1*n+i];
        pos[(size_t)2*n_ptcl + dst] = sc->pos[2*n+i];
        mm[dst]     = sc->mm[i];
        orgind[dst] = sc->org[i];
    }
}

/* =========================================================
 *  Build a single KD-tree node (recursive).
 *  sc: per-thread scratch buffer (must not be shared across threads).
 * ========================================================= */
static void build_node(Node * restrict nodes,
                       double * restrict pos, double * restrict mm,
                       int * restrict orgind,
                       int bsize, int *numnode,
                       int bstart, int bend, int level, int n_ptcl,
                       Scratch *sc)
{
    int nid = ++(*numnode);
    Node *nd = &nodes[nid];
    nd->id = nid; nd->bstart = bstart; nd->bend = bend;
    nd->ncount = bend - bstart + 1; nd->level = level; nd->leaf = -1;
    nd->left = nd->right = nd->sibling = nd->parent = 0;

    int n = nd->ncount;

    /* Centre of mass */
    double total_m = 0.0;
    for (int i = bstart; i <= bend; i++) total_m += mm[i];

    if (total_m > 0.0) {
        double inv_m = 1.0 / total_m;
        double c0=0, c1=0, c2=0;
        for (int i = bstart; i <= bend; i++) {
            double mi = mm[i];
            c0 += POS(i,0)*mi; c1 += POS(i,1)*mi; c2 += POS(i,2)*mi;
        }
        nd->cen[0]=c0*inv_m; nd->cen[1]=c1*inv_m; nd->cen[2]=c2*inv_m;
    } else {
        double inv_n = 1.0 / n, c0=0, c1=0, c2=0;
        for (int i = bstart; i <= bend; i++) {
            c0 += POS(i,0); c1 += POS(i,1); c2 += POS(i,2);
        }
        nd->cen[0]=c0*inv_n; nd->cen[1]=c1*inv_n; nd->cen[2]=c2*inv_n;
    }

    if (n <= bsize) { nd->mass = total_m; nd->leaf = 1; return; }

    /* Split dim: widest range */
    int sdim = 0;
    double smax = nd->bnd[0][1] - nd->bnd[0][0];
    for (int k = 1; k < 3; k++) {
        double rng = nd->bnd[k][1] - nd->bnd[k][0];
        if (rng > smax) { smax = rng; sdim = k; }
    }
    nd->splitdim = sdim;

    for (int i = 0; i < n; i++) { sc->key[i]=POS(bstart+i,sdim); sc->perm[i]=i; }
    int med = n / 2;
    nth_element(sc->key, sc->perm, 0, n-1, med);
    nd->splitval = sc->key[med];
    nd->splitind = bstart + med + 1;
    reorder_seg(pos, mm, orgind, sc->perm, n, bstart, n_ptcl, sc);

    int cur = nid;
    nodes[cur].left            = *numnode + 1;
    nodes[*numnode+1].parent   = cur;
    memcpy(nodes[*numnode+1].bnd, nodes[cur].bnd, sizeof(nodes[0].bnd));
    nodes[*numnode+1].bnd[sdim][1] = nodes[cur].splitval;
    build_node(nodes, pos, mm, orgind, bsize, numnode,
               bstart, bstart+med, level+1, n_ptcl, sc);

    nodes[cur].right           = *numnode + 1;
    nodes[*numnode+1].parent   = cur;
    memcpy(nodes[*numnode+1].bnd, nodes[cur].bnd, sizeof(nodes[0].bnd));
    nodes[*numnode+1].bnd[sdim][0] = nodes[cur].splitval;
    build_node(nodes, pos, mm, orgind, bsize, numnode,
               bstart+med+1, bend, level+1, n_ptcl, sc);

    nodes[nodes[cur].left].sibling  = nodes[cur].right;
    nodes[nodes[cur].right].sibling = nodes[cur].left;
}

/* =========================================================
 *  2-phase parallel tree build:
 *    Phase 1 (sequential): build coarse tree until act_thread leaves
 *    Phase 2 (OMP parallel): build each subtree independently
 *  Returns total numnode.  nodes[] must be 1-indexed.
 * ========================================================= */
static int build_tree_parallel(Node * restrict nodes,
                               double * restrict pos,
                               double * restrict mm,
                               int * restrict orgind,
                               int bsize, int n_ptcl, int n_thread)
{
    /* Largest power of 2 <= n_thread */
    int act = 1;
    while (act * 2 <= n_thread) act *= 2;

    /* Phase 1: coarse tree with act leaf seeds */
    Scratch sc0 = scratch_alloc(n_ptcl);
    int coarse_bsize = n_ptcl / act + 1;
    int numnode = 0;
    for (int k = 0; k < 3; k++) {
        double lo=POS(0,k), hi=POS(0,k);
        for (int i=1; i<n_ptcl; i++) {
            double v=POS(i,k);
            if(v<lo)lo=v; if(v>hi)hi=v;
        }
        nodes[1].bnd[k][0]=lo; nodes[1].bnd[k][1]=hi;
    }
    build_node(nodes, pos, mm, orgind, coarse_bsize, &numnode,
               0, n_ptcl-1, 1, n_ptcl, &sc0);
    scratch_free(&sc0);

    int n_ini = numnode;

    /* If already leaf-sized, nothing more to do */
    if (coarse_bsize <= bsize) { nodes[1].numnode = numnode; return numnode; }

    /* Collect coarse leaves (the act seeds).
     * Due to the asymmetric split (left gets n/2+1 particles when n is even),
     * the coarse tree can produce slightly more than `act` leaves.
     * Allocate generously: 2*act+16 covers all realistic cases. */
    Node *seeds = (Node *)malloc((2*act + 16) * sizeof(Node));
    {
        int k = 0;
        for (int i = 1; i <= n_ini; i++)
            if (nodes[i].leaf == 1) seeds[k++] = nodes[i];
        act = k;
    }

    /* Upper bound for nodes in each subtree.
     * Must be based on coarse_bsize (the max seed size), NOT n_ptcl/act.
     * After the coarse build, act may have grown beyond the original power-of-2
     * value, but some seeds can still hold up to coarse_bsize particles.
     * Using n_ptcl/act (the new, larger act) underestimates those seeds and
     * causes a buffer overflow in sub_nodes[t]. */
    int sub_max = (bsize >= 2) ? 4*(coarse_bsize/bsize + 2) + 16 : 2*coarse_bsize+4;

    /* Per-thread subtree node arrays */
    Node **sub_nodes = (Node **)malloc(act * sizeof(Node *));
    int   *sub_nums  = (int   *)malloc(act * sizeof(int));
    for (int i = 0; i < act; i++) {
        sub_nodes[i] = (Node *)calloc((size_t)(sub_max+2), sizeof(Node));
        sub_nums[i]  = 0;
    }

    /* Phase 2: parallel subtree build */
    #pragma omp parallel for schedule(dynamic,1) num_threads(act)
    for (int t = 0; t < act; t++) {
        int bs = seeds[t].bstart;
        int be = seeds[t].bend;
        int nsub = be - bs + 1;

        /* Each thread gets its own scratch (avoids false sharing) */
        Scratch sc = scratch_alloc(nsub);

        /* Root of this subtree shares the same bnd as the coarse seed */
        memcpy(sub_nodes[t][1].bnd, seeds[t].bnd, sizeof(seeds[t].bnd));

        int nn = 0;
        build_node(sub_nodes[t], pos, mm, orgind, bsize, &nn,
                   bs, be, seeds[t].level+1, n_ptcl, &sc);
        sub_nums[t] = nn;

        scratch_free(&sc);
    }

    /* ---- Merge subtrees into the main nodes[] array ---- */
    /* Total size needed */
    int n_aft = 0;
    for (int t = 0; t < act; t++) n_aft += sub_nums[t];

    /*
     * ---- Merge subtrees into nodes[] ----
     *
     * ID offset rule (Fortran-equivalent):
     *   id_off starts at n_ini - 1
     *   subtree-local node i  →  global position (i + id_off)  == its new ID
     *   After subtree t:  id_off += nnums[t] - 1
     *
     * Virtual root (node 1 of each subtree) covers the same range as the
     * seed — it is NOT copied; instead the seed's left/right are updated
     * to point to node 1's children (stored at id_off+lchild, id_off+rchild).
     */
    int id_off = n_ini - 1;

    for (int t = 0; t < act; t++) {
        int nn = sub_nums[t];
        Node *sn = sub_nodes[t];
        int sid = seeds[t].id;

        /* Save virtual-root's children before we touch sn[] */
        int lchild = sn[1].left;
        int rchild = sn[1].right;

        /* Adjust nodes 2..nn and place each at its global position */
        for (int i = 2; i <= nn; i++) {
            Node nd = sn[i];
            nd.id      = i + id_off;
            nd.parent  = (nd.parent  == 1) ? sid               /* direct child of vroot → seed */
                       : (nd.parent  >  0) ? nd.parent  + id_off : 0;
            nd.left    = (nd.left    >  0) ? nd.left    + id_off : 0;
            nd.right   = (nd.right   >  0) ? nd.right   + id_off : 0;
            nd.sibling = (nd.sibling >  0) ? nd.sibling + id_off : 0;
            nodes[nd.id] = nd;   /* position == ID */
        }

        /* Connect seed to virtual-root's children */
        if (nn > 1 && lchild > 0) {
            nodes[sid].leaf  = -1;
            nodes[sid].left  = lchild + id_off;
            nodes[sid].right = rchild + id_off;
        } else {
            /* Subtree collapsed to a single leaf — update seed in place */
            nodes[sid].leaf = 1;
            nodes[sid].mass = sn[1].mass;
            memcpy(nodes[sid].cen, sn[1].cen, sizeof(sn[1].cen));
        }

        id_off += nn - 1;
        free(sub_nodes[t]);
    }

    free(sub_nodes); free(sub_nums); free(seeds);
    numnode = id_off + 1;
    nodes[1].numnode = numnode;
    return numnode;
}

/* =========================================================
 *  Trilinear interpolation (exact match to Fortran).
 *  Corner layout bnd[c][k]:
 *    0:(xlo,ylo,zlo) 1:(xlo,yhi,zlo) 2:(xhi,yhi,zlo) 3:(xhi,ylo,zlo)
 *    4:(xlo,ylo,zhi) 5:(xlo,yhi,zhi) 6:(xhi,yhi,zhi) 7:(xhi,ylo,zhi)
 * ========================================================= */
static inline double lint(double x, double x0, double x1, double v0, double v1) {
    double dx = (x1==x0) ? 0.0 : (x-x0)/(x1-x0);
    if (dx < 0.0) dx = 0.0;
    return dx*v1 + (1.0-dx)*v0;
}

static void getbnd(const Node * restrict nd, double bnd[8][3]) {
    double xlo=nd->bnd[0][0], xhi=nd->bnd[0][1];
    double ylo=nd->bnd[1][0], yhi=nd->bnd[1][1];
    double zlo=nd->bnd[2][0], zhi=nd->bnd[2][1];
    bnd[0][0]=xlo; bnd[0][1]=ylo; bnd[0][2]=zlo;
    bnd[1][0]=xlo; bnd[1][1]=yhi; bnd[1][2]=zlo;
    bnd[2][0]=xhi; bnd[2][1]=yhi; bnd[2][2]=zlo;
    bnd[3][0]=xhi; bnd[3][1]=ylo; bnd[3][2]=zlo;
    bnd[4][0]=xlo; bnd[4][1]=ylo; bnd[4][2]=zhi;
    bnd[5][0]=xlo; bnd[5][1]=yhi; bnd[5][2]=zhi;
    bnd[6][0]=xhi; bnd[6][1]=yhi; bnd[6][2]=zhi;
    bnd[7][0]=xhi; bnd[7][1]=ylo; bnd[7][2]=zhi;
}

static inline double interpole(const double bnd[8][3],
                                const double p[3], const double v[8]) {
    double v1=lint(p[1],bnd[0][1],bnd[1][1],v[0],v[1]);
    double v2=lint(p[1],bnd[3][1],bnd[2][1],v[3],v[2]);
    double v0=lint(p[0],bnd[0][0],bnd[2][0],v1,  v2 );
    double w1=lint(p[1],bnd[4][1],bnd[5][1],v[4],v[5]);
    double w2=lint(p[1],bnd[7][1],bnd[6][1],v[7],v[6]);
    double w0=lint(p[0],bnd[4][0],bnd[6][0],w1,  w2 );
    return lint(p[2],bnd[0][2],bnd[4][2],v0,w0);
}

/* =========================================================
 *  Main entry point
 * ========================================================= */
void js_getpt_ft(int argc, void *argv[])
{
    int    *larr  = (int    *)argv[0];
    double *darr  = (double *)argv[1];
    double *pos   = (double *)argv[2];
    double *mm    = (double *)argv[3];
    double *pot   = (double *)argv[4];
    double *force = (double *)argv[5];

    int    n_ptcl   = larr[0];
    int    n_thread = larr[2];
    int    p_type   = larr[3];
    int    e_type   = larr[4];
    int    bsize    = larr[12];
    double Gconst   = darr[0];
    double negG     = -Gconst;

    if (n_ptcl <= 0) return;
    omp_set_num_threads(n_thread);

    memset(pot,   0, (size_t)n_ptcl * sizeof(double));
    memset(force, 0, (size_t)n_ptcl * sizeof(double));

    int *orgind = (int *)malloc(n_ptcl * sizeof(int));
    for (int i = 0; i < n_ptcl; i++) orgind[i] = i+1;

    int max_nodes = (bsize >= 2) ? 4*(n_ptcl/bsize + 4) + 32 : 2*n_ptcl + 8;
    /* Extra headroom for the parallel merge step */
    max_nodes = (int)(max_nodes * 1.5) + n_thread * 32;
    Node *nodes = (Node *)calloc((size_t)(max_nodes+2), sizeof(Node));

    double t0 = omp_get_wtime();

    int numnode;
    if (n_thread > 1 && n_ptcl / n_thread > bsize * 2) {
        /* Parallel build only when subtrees are meaningfully larger than bsize */
        numnode = build_tree_parallel(nodes, pos, mm, orgind,
                                      bsize, n_ptcl, n_thread);
    } else {
        /* Sequential build */
        Scratch sc = scratch_alloc(n_ptcl);
        for (int k = 0; k < 3; k++) {
            double lo=POS(0,k), hi=POS(0,k);
            for (int i=1; i<n_ptcl; i++) {
                double v=POS(i,k); if(v<lo)lo=v; if(v>hi)hi=v;
            }
            nodes[1].bnd[k][0]=lo; nodes[1].bnd[k][1]=hi;
        }
        numnode = 0;
        build_node(nodes, pos, mm, orgind, bsize, &numnode,
                   0, n_ptcl-1, 1, n_ptcl, &sc);
        nodes[1].numnode = numnode;
        scratch_free(&sc);
    }

    double t1 = omp_get_wtime();

    /* ---- Collect leaves — SoA layout for cache-friendly inner loops ---- */
    int n_leaf = 0;
    for (int i = 1; i <= numnode; i++) if (nodes[i].leaf==1) n_leaf++;

    double *lcx   = (double *)malloc(n_leaf * sizeof(double));
    double *lcy   = (double *)malloc(n_leaf * sizeof(double));
    double *lcz   = (double *)malloc(n_leaf * sizeof(double));
    double *lmass = (double *)malloc(n_leaf * sizeof(double));
    int    *lbs   = (int    *)malloc(n_leaf * sizeof(int));
    int    *lbe   = (int    *)malloc(n_leaf * sizeof(int));

    /* ptcl2leaf[i] = leaf index containing tree-position particle i */
    int *ptcl2leaf = (int *)malloc(n_ptcl * sizeof(int));

    /* lbnd only needed for e_type=0 */
    double (*lbnd)[8][3] = NULL;
    if (e_type == 0)
        lbnd = (double (*)[8][3])malloc((size_t)n_leaf * sizeof(*lbnd));

    {
        int j = 0;
        for (int i = 1; i <= numnode; i++) {
            if (nodes[i].leaf != 1) continue;
            lcx[j]   = nodes[i].cen[0];
            lcy[j]   = nodes[i].cen[1];
            lcz[j]   = nodes[i].cen[2];
            lmass[j] = nodes[i].mass;
            lbs[j]   = nodes[i].bstart;
            lbe[j]   = nodes[i].bend;
            for (int k = lbs[j]; k <= lbe[j]; k++) ptcl2leaf[k] = j;
            if (lbnd) getbnd(&nodes[i], lbnd[j]);
            j++;
        }
    }

    /* =====================================================
     * Potential computation
     * ===================================================== */

    if (e_type == 1) {
        /*
         * Exact mode.  p_type=0 restructured for branch-free inner loop:
         *   pot += sum_all_leaves negG*mass/r   (vectorizable)
         *   pot -= negG*mm[i]/r_to_own_center   (single correction)
         *
         * p_type=1: monopole outside own leaf, direct within own leaf.
         */
        #pragma omp parallel for schedule(static) default(shared)
        for (int i = 0; i < n_ptcl; i++) {
            double xi=POS(i,0), yi=POS(i,1), zi=POS(i,2);
            double pi=0.0, fi=0.0;

            if (p_type == 0) {
                /* Branch-free inner loop — auto-vectorizable with -O3 -march=native */
                #pragma omp simd reduction(+:pi,fi)
                for (int j = 0; j < n_leaf; j++) {
                    double dx2 = (lcx[j]-xi)*(lcx[j]-xi)
                               + (lcy[j]-yi)*(lcy[j]-yi)
                               + (lcz[j]-zi)*(lcz[j]-zi);
                    double inv_d = (dx2 > 0.0) ? 1.0/sqrt(dx2) : 0.0;
                    pi += negG * lmass[j] * inv_d;
                    fi += negG * lmass[j] * inv_d * inv_d;
                }
                /* Subtract own mass at own leaf centre (one correction) */
                int jj = ptcl2leaf[i];
                double dx2 = (lcx[jj]-xi)*(lcx[jj]-xi)
                           + (lcy[jj]-yi)*(lcy[jj]-yi)
                           + (lcz[jj]-zi)*(lcz[jj]-zi);
                if (dx2 > 0.0) {
                    double inv_d = 1.0/sqrt(dx2);
                    pi -= negG * mm[i] * inv_d;
                    fi -= negG * mm[i] * inv_d * inv_d;
                }
            } else {
                /* p_type=1: direct within own leaf, monopole outside */
                int my = ptcl2leaf[i];
                for (int j = 0; j < n_leaf; j++) {
                    if (j == my) {
                        for (int k=lbs[j]; k<=lbe[j]; k++) {
                            if (k==i) continue;
                            double dx2 = (POS(k,0)-xi)*(POS(k,0)-xi)
                                       + (POS(k,1)-yi)*(POS(k,1)-yi)
                                       + (POS(k,2)-zi)*(POS(k,2)-zi);
                            if (dx2==0.0) continue;
                            double inv_d = 1.0/sqrt(dx2);
                            pi += negG*mm[k]*inv_d;
                            fi += negG*mm[k]*inv_d*inv_d;
                        }
                    } else {
                        double dx2 = (lcx[j]-xi)*(lcx[j]-xi)
                                   + (lcy[j]-yi)*(lcy[j]-yi)
                                   + (lcz[j]-zi)*(lcz[j]-zi);
                        if (dx2==0.0) continue;
                        double inv_d = 1.0/sqrt(dx2);
                        pi += negG*lmass[j]*inv_d;
                        fi += negG*lmass[j]*inv_d*inv_d;
                    }
                }
            }
            pot[i]=pi; force[i]=fi;
        }

    } else {
        /*
         * Fast mode (e_type=0): 8-corner trilinear interpolation.
         * Restructured to include self-leaf in the inner loop (branch-free),
         * then subtract self-contribution after.
         * Each leaf writes to disjoint particle ranges — no OMP race.
         */
        #pragma omp parallel for schedule(dynamic,4) default(shared)
        for (int i = 0; i < n_leaf; i++) {
            double dv[8]={0,0,0,0,0,0,0,0};
            double df[8]={0,0,0,0,0,0,0,0};

            /* Sum ALL leaves (including self).
             * Loop order: c outside, j inside — the long j loop (n_leaf)
             * becomes a simple reduction and is vectorized by the compiler. */
            for (int c = 0; c < 8; c++) {
                double bx = lbnd[i][c][0], by = lbnd[i][c][1], bz = lbnd[i][c][2];
                double sumv = 0.0, sumf = 0.0;
                #pragma omp simd reduction(+:sumv,sumf)
                for (int j = 0; j < n_leaf; j++) {
                    double dx = lcx[j]-bx, dy = lcy[j]-by, dz = lcz[j]-bz;
                    double dx2 = dx*dx + dy*dy + dz*dz;
                    double inv_d = (dx2 > 0.0) ? 1.0/sqrt(dx2) : 0.0;
                    sumv += negG*lmass[j]*inv_d;
                    sumf += negG*lmass[j]*inv_d*inv_d;
                }
                dv[c] = sumv;
                df[c] = sumf;
            }
            /* Subtract own-leaf centre contribution */
            for (int c = 0; c < 8; c++) {
                double dx2 = (lcx[i]-lbnd[i][c][0])*(lcx[i]-lbnd[i][c][0])
                           + (lcy[i]-lbnd[i][c][1])*(lcy[i]-lbnd[i][c][1])
                           + (lcz[i]-lbnd[i][c][2])*(lcz[i]-lbnd[i][c][2]);
                double inv_d = (dx2 > 0.0) ? 1.0/sqrt(dx2) : 0.0;
                dv[c] -= negG*lmass[i]*inv_d;
                df[c] -= negG*lmass[i]*inv_d*inv_d;
            }

            int bs=lbs[i], be=lbe[i];
            for (int j=bs; j<=be; j++) {
                double pj[3]={POS(j,0),POS(j,1),POS(j,2)};

                pot[j]   += interpole(lbnd[i], pj, dv);
                force[j] += interpole(lbnd[i], pj, df);

                if (p_type == 0) {
                    double dx2 = (lcx[i]-pj[0])*(lcx[i]-pj[0])
                               + (lcy[i]-pj[1])*(lcy[i]-pj[1])
                               + (lcz[i]-pj[2])*(lcz[i]-pj[2]);
                    if (dx2==0.0) continue;
                    double inv_d = 1.0/sqrt(dx2);
                    pot[j]   += negG*(lmass[i]-mm[j])*inv_d;
                    force[j] += negG*(lmass[i]-mm[j])*inv_d*inv_d;
                } else {
                    for (int k=bs; k<=be; k++) {
                        if (k==j) continue;
                        double dx2 = (POS(k,0)-pj[0])*(POS(k,0)-pj[0])
                                   + (POS(k,1)-pj[1])*(POS(k,1)-pj[1])
                                   + (POS(k,2)-pj[2])*(POS(k,2)-pj[2]);
                        if (dx2==0.0) continue;
                        double inv_d = 1.0/sqrt(dx2);
                        pot[j]   += negG*mm[k]*inv_d;
                        force[j] += negG*mm[k]*inv_d*inv_d;
                    }
                }
            }
        }
    }

    double t2 = omp_get_wtime();

    /* ---- Reorder back to original particle order ---- */
    /* orgind[tree_pos] = 1-indexed original ID; scatter directly — O(N), no recursion */
    double *tmp_pot   = (double *)malloc(n_ptcl * sizeof(double));
    double *tmp_force = (double *)malloc(n_ptcl * sizeof(double));
    for (int j = 0; j < n_ptcl; j++) {
        tmp_pot  [orgind[j]-1] = pot  [j];
        tmp_force[orgind[j]-1] = force[j];
    }
    memcpy(pot,   tmp_pot,   (size_t)n_ptcl*sizeof(double));
    memcpy(force, tmp_force, (size_t)n_ptcl*sizeof(double));

    (void)t0; (void)t1; (void)t2;  /* suppress unused-variable warnings */

    free(tmp_pot); free(tmp_force);
    free(ptcl2leaf); free(lbnd);
    free(lcx); free(lcy); free(lcz); free(lmass); free(lbs); free(lbe);
    free(nodes); free(orgind);
}

/* =========================================================
 *  Clean C wrapper — callable from C++
 * ========================================================= */
#include "getpt.h"
#include <omp.h>

void run_bh_potential(int n_ptcl,
                      const double *x, const double *y, const double *z,
                      const double *mass,
                      double *pot,
                      int n_thread,
                      int bsize)
{
    if (n_ptcl <= 0) return;

    if (n_thread <= 0) n_thread = omp_get_max_threads();
    if (bsize <= 0)    bsize = 512;

    /* Reduce bsize until 8*bsize <= n_ptcl (mirrors IDL code) */
    while (n_ptcl < (long)8 * bsize && bsize > 4) bsize /= 2;

    /* Build column-major position array (x first, then y, then z) */
    double *pos   = (double *)malloc((size_t)n_ptcl * 3 * sizeof(double));
    double *mm    = (double *)malloc((size_t)n_ptcl * sizeof(double));
    double *force = (double *)malloc((size_t)n_ptcl * sizeof(double));
    if (!pos || !mm || !force) {
        free(pos); free(mm); free(force);
        return;
    }

    for (int i = 0; i < n_ptcl; i++) {
        pos[(size_t)0 * n_ptcl + i] = x[i];
        pos[(size_t)1 * n_ptcl + i] = y[i];
        pos[(size_t)2 * n_ptcl + i] = z[i];
        mm[i]  = mass[i];
        pot[i] = 0.0;
    }

    /* G in (km/s)^2 kpc/Msun */
    double Gconst = 6.67408e-11 * (1.0/3.086e19) / (1.0/1.98892e30) * 1e-6;

    int    larr[20] = {0};
    double darr[20] = {0.0};
    larr[0]  = n_ptcl;
    larr[1]  = 3;           /* 3-D */
    larr[2]  = n_thread;
    larr[3]  = 0;           /* p_type = mesh */
    larr[4]  = 0;           /* e_type = pole */
    larr[10] = 0;
    larr[11] = 0;
    larr[12] = bsize;
    darr[0]  = Gconst;

    void *argv[6] = {larr, darr, pos, mm, pot, force};
    js_getpt_ft(6, argv);

    free(pos); free(mm); free(force);
}
