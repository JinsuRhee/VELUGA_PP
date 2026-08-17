// bprop.cxx — bulk photometric property computation (rv_BProp)
//
// Unified per-chunk streaming: both newramses=0 and =1 use the same lambdas.
//
// Stars (bprop, horg='g'):
//   - Hash-based member matching in O(N_chunk) per chunk.
//   - Per-chunk kd-tree for spatial (finite-aperture) queries.
//   - Galaxy "done" tracking: once all member particles are found, galaxy is
//     excluded from future member matching.  When no finite apertures are
//     configured, streaming stops as soon as every galaxy is done.
//
// DM (confrac, both horg):
//   - Per-chunk kd-tree with chunk-AABB early-exit: chunks that do not
//     overlap any galaxy's confrac sphere are skipped entirely.

#include "pipeline/bprop.h"
#include "pipeline/rawdata.h"
#include "pipeline/kdtree.h"
#include "pipeline/cosmo.h"
#include "pipeline/miles.h"

#ifdef VPP_USE_OMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace Pipeline {

namespace {
static bool has_col(const RawCatalog &cat, const std::string &name)
{
    return cat.f64.count(name) > 0;
}
} // namespace

// ===========================================================================
// compute_bprop
// ===========================================================================
BPropResult compute_bprop(
    const vpp_set::Settings  &vh,
    int32_t                   snap,
    const RawCatalog         &cat,
    const PtclIDCatalog      &ptcl,
    const CosmoInfo          &cosmo,
    const MilesML            &miles,
    const CosmTable          &ctbl)
{
    BPropResult res;

    const int64_t n_gal     = cat.n;
    const int32_t n_flux    = (int32_t)vh.flux_list.size();
    const int32_t n_mag_ap  = (int32_t)vh.mag_r.size();
    const int32_t n_sfr     = (int32_t)vh.sfr_r.size();
    const int32_t n_conf_ap = (int32_t)vh.conf_r.size();
    const double  NaN       = std::numeric_limits<double>::quiet_NaN();

    res.n_gal = n_gal; res.n_flux = n_flux; res.n_mag_ap = n_mag_ap;
    res.n_sfr = n_sfr; res.n_conf_ap = n_conf_ap;
    if (n_mag_ap > 0 && n_flux > 0) {
        res.abmag.assign((size_t)(n_gal*n_mag_ap*n_flux), NaN);
        res.sbf  .assign((size_t)(n_gal*n_mag_ap*n_flux), NaN);
    }
    if (n_sfr > 0) res.sfr.assign((size_t)(n_gal*n_sfr), NaN);
    res.isclump.assign((size_t)n_gal, -1);
    if (n_conf_ap > 0) {
        res.confrac_n.assign((size_t)(n_gal*n_conf_ap), NaN);
        res.confrac_m.assign((size_t)(n_gal*n_conf_ap), NaN);
    }

    if (n_gal == 0) return res;

    int nthreads = 1;
#ifdef VPP_USE_OMP
    nthreads = omp_get_max_threads();
#endif

    // HDF5 part filename (newramses=1)
    std::string part_fn;
    if (vh.newramses != 0) {
        std::ostringstream oss;
        oss << vh.dir_raw << "/part_"
            << std::setw(5) << std::setfill('0') << snap << ".h5";
        part_fn = oss.str();
    }

    // ========================================================================
    // Shared galaxy geometry (code units)
    // ========================================================================
    const double inv_kpc = 1.0 / cosmo.kpc_per_code;
    const bool have_xyz   = has_col(cat,"Xc") && has_col(cat,"Yc") && has_col(cat,"Zc");
    const bool have_rhalf = has_col(cat,"R_HalfMass");

    std::vector<double> gal_xc(n_gal,0.0), gal_yc(n_gal,0.0), gal_zc(n_gal,0.0);
    std::vector<double> gal_rhalf(n_gal,0.0);

    if (have_xyz) {
        const auto &cxv=cat.f64.at("Xc"), &cyv=cat.f64.at("Yc"), &czv=cat.f64.at("Zc");
        for (int64_t gi=0; gi<n_gal; ++gi) {
            gal_xc[gi] = cxv[(size_t)gi]*inv_kpc;
            gal_yc[gi] = cyv[(size_t)gi]*inv_kpc;
            gal_zc[gi] = czv[(size_t)gi]*inv_kpc;
        }
    }
    if (have_rhalf) {
        const auto &crh = cat.f64.at("R_HalfMass");
        for (int64_t gi=0; gi<n_gal; ++gi)
            gal_rhalf[gi] = crh[(size_t)gi]*inv_kpc;
    }

    // ========================================================================
    // BPROP SETUP (horg='g')
    // ========================================================================
    const bool want_bprop = !vh.skip_bprop && vh.horg=='g'
                            && n_flux>0 && (n_mag_ap>0||n_sfr>0)
                            && have_xyz && have_rhalf;
    bool do_bprop = false;
    bool has_finite_ap = false;

    // Member map: star_id → galaxy index
    std::unordered_map<int64_t,int64_t> member_map;

    // Galaxy "done" tracking (member matching)
    std::vector<int64_t> total_members(n_gal, 0);
    std::vector<int64_t> found_total  (n_gal, 0);
    std::vector<bool>    gal_done     (n_gal, false);
    int64_t n_done = 0;

    // Spatial aperture geometry
    std::vector<double> gal_max_r(n_gal, 0.0);  // max finite aperture radius (code units)
    double sp_xlo=1e30, sp_xhi=-1e30, sp_ylo=1e30, sp_yhi=-1e30, sp_zlo=1e30, sp_zhi=-1e30;

    // Aperture type flags
    std::vector<bool> mag_use_mb, sfr_use_mb;

    // Thread accumulators
    size_t szlum=0, szsfr=0;
    std::vector<std::vector<double>> th_mb_lum, th_mb_sfr, th_mb_mass;
    std::vector<std::vector<double>> th_sp_lum, th_sp_sfr;
    std::vector<std::vector<int64_t>> th_found;   // per-thread, per-chunk found count

    // Unit conversions
    double kpc=0.0, msun=0.0, z_snap=0.0;
    std::vector<int32_t> band_idx;

    // Clump
    bool do_clump = false;
    static const std::vector<double> _empty_dv;
    const std::vector<double> *cat_mtot_ptr = &_empty_dv;

    // Age buffer (reused across chunks)
    std::vector<float> age_gyr_buf;

    if (want_bprop) {
        kpc    = cosmo.kpc_per_code;
        msun   = cosmo.Msun_per_code;
        z_snap = 1.0/cosmo.aexp - 1.0;

        band_idx.resize((size_t)n_flux);
        for (int32_t fi=0; fi<n_flux; ++fi) {
            band_idx[(size_t)fi] = band_index(vh.flux_list[(size_t)fi]);
            if (band_idx[(size_t)fi] < 0)
                LOG() << "[bprop] WARNING: unknown band '" << vh.flux_list[(size_t)fi] << "'";
        }

        const bool have_mtot = has_col(cat,"Mass_tot");
        const bool dc2 = vh.pp_clump_mfrac>0.0 && n_sfr>0 && have_mtot;
        do_clump = dc2 && vh.sfr_t[0]>0.0;
        if (vh.pp_clump_mfrac>0.0 && n_sfr>0 && !have_mtot)
            LOG() << "[bprop] WARNING: 'Mass_tot' missing — clump disabled";
        if (do_clump) cat_mtot_ptr = &cat.f64.at("Mass_tot");

        mag_use_mb.assign((size_t)n_mag_ap, false);
        sfr_use_mb.assign((size_t)n_sfr,    false);
        for (int32_t ai=0; ai<n_mag_ap; ++ai) mag_use_mb[(size_t)ai] = (vh.mag_r[(size_t)ai]<0.0);
        for (int32_t si=0; si<n_sfr;    ++si) sfr_use_mb[(size_t)si] = (vh.sfr_r[(size_t)si]<0.0);

        for (int32_t ai=0; ai<n_mag_ap; ++ai) if (!mag_use_mb[(size_t)ai]) { has_finite_ap=true; break; }
        if (!has_finite_ap)
            for (int32_t si=0; si<n_sfr; ++si) if (!sfr_use_mb[(size_t)si]) { has_finite_ap=true; break; }

        // Per-galaxy max spatial aperture radius and global AABB
        for (int64_t gi=0; gi<n_gal; ++gi) {
            double mr=0.0;
            for (int32_t ai=0; ai<n_mag_ap; ++ai) if (!mag_use_mb[(size_t)ai]) mr=std::max(mr,vh.mag_r[(size_t)ai]*gal_rhalf[(size_t)gi]);
            for (int32_t si=0; si<n_sfr;    ++si) if (!sfr_use_mb[(size_t)si]) mr=std::max(mr,vh.sfr_r[(size_t)si]*gal_rhalf[(size_t)gi]);
            gal_max_r[(size_t)gi] = mr;
        }
        if (has_finite_ap) {
            for (int64_t gi=0; gi<n_gal; ++gi) {
                const double r = gal_max_r[(size_t)gi];
                sp_xlo=std::min(sp_xlo,gal_xc[(size_t)gi]-r); sp_xhi=std::max(sp_xhi,gal_xc[(size_t)gi]+r);
                sp_ylo=std::min(sp_ylo,gal_yc[(size_t)gi]-r); sp_yhi=std::max(sp_yhi,gal_yc[(size_t)gi]+r);
                sp_zlo=std::min(sp_zlo,gal_zc[(size_t)gi]-r); sp_zhi=std::max(sp_zhi,gal_zc[(size_t)gi]+r);
            }
        }

        // Member map + total counts
        member_map.reserve((size_t)((ptcl.n_bdn+ptcl.n_ubd)*1.3+1));
        for (int64_t gi=0; gi<n_gal; ++gi) {
            for (int64_t k=ptcl.b_ind[(size_t)(gi*2)]; k<=ptcl.b_ind[(size_t)(gi*2+1)]; ++k)
                member_map[ptcl.p_id[(size_t)k]] = gi;
            for (int64_t k=ptcl.u_ind[(size_t)(gi*2)]; k<=ptcl.u_ind[(size_t)(gi*2+1)]; ++k)
                member_map[ptcl.p_id[(size_t)k]] = gi;
            const int64_t nb = ptcl.b_ind[(size_t)(gi*2+1)] >= ptcl.b_ind[(size_t)(gi*2)]
                               ? ptcl.b_ind[(size_t)(gi*2+1)] - ptcl.b_ind[(size_t)(gi*2)] + 1 : 0;
            const int64_t nu = ptcl.u_ind[(size_t)(gi*2+1)] >= ptcl.u_ind[(size_t)(gi*2)]
                               ? ptcl.u_ind[(size_t)(gi*2+1)] - ptcl.u_ind[(size_t)(gi*2)] + 1 : 0;
            total_members[(size_t)gi] = nb + nu;
        }
        LOG() << "[bprop] member_map: " << member_map.size() << " star entries";

        szlum = (size_t)(n_gal*n_mag_ap*n_flux);
        szsfr = (size_t)(n_gal*n_sfr);
        th_mb_lum .assign((size_t)nthreads, std::vector<double>(szlum, 0.0));
        th_mb_sfr .assign((size_t)nthreads, std::vector<double>(szsfr, 0.0));
        th_mb_mass.assign((size_t)nthreads, std::vector<double>((size_t)n_gal, 0.0));
        th_sp_lum .assign((size_t)nthreads, std::vector<double>(szlum, 0.0));
        th_sp_sfr .assign((size_t)nthreads, std::vector<double>(szsfr, 0.0));
        th_found  .assign((size_t)nthreads, std::vector<int64_t>((size_t)n_gal, 0LL));

        do_bprop = true;
    } else if (!vh.skip_bprop && vh.horg=='g' && n_flux>0 && (n_mag_ap>0||n_sfr>0)) {
        LOG() << "[bprop] WARNING: Xc/Yc/Zc or R_HalfMass missing — skipping photometry";
    }

    // ========================================================================
    // CONFRAC SETUP
    // ========================================================================
    const bool want_confrac = !vh.skip_confrac && n_conf_ap>0 && have_xyz;
    bool do_confrac = false;

    std::vector<double> gal_rap (n_gal*n_conf_ap, 0.0);
    std::vector<double> gal_rap2(n_gal*n_conf_ap, 0.0);
    std::vector<double> gal_rmax_c(n_gal, 0.0);
    double cf_xlo=1e30, cf_xhi=-1e30, cf_ylo=1e30, cf_yhi=-1e30, cf_zlo=1e30, cf_zhi=-1e30;
    float  lr_thresh = 0.0f;
    size_t sz_conf = 0;
    std::vector<std::vector<int64_t>> th_ntot, th_nlr;
    std::vector<std::vector<double>>  th_mtot, th_mlr;

    if (want_confrac) {
        const std::string rref_col = (vh.horg=='g') ? "R_HalfMass" : "R_200crit";
        if (!has_col(cat, rref_col)) {
            LOG() << "[bprop] ERROR: '" << rref_col << "' missing — skipping confrac";
        } else {
            if (vh.neff>0 && cosmo.omega_m>0.0) {
                const double neff3 = (double)vh.neff*vh.neff*vh.neff;
                const double dmc   = (cosmo.omega_m-cosmo.omega_b)/(cosmo.omega_m*neff3);
                lr_thresh = (float)(dmc*2.0);
                LOG() << "[bprop] dmp_mass(code)=" << dmc << "  lr_thresh=" << lr_thresh;
            } else {
                LOG() << "[bprop] WARNING: neff not set — confrac lr_thresh=0";
            }

            const auto &crref = cat.f64.at(rref_col);
            for (int64_t gi=0; gi<n_gal; ++gi) {
                const double rref = crref[(size_t)gi]*inv_kpc;
                double rm = 0.0;
                for (int32_t ai=0; ai<n_conf_ap; ++ai) {
                    const double r = vh.conf_r[(size_t)ai]*rref;
                    gal_rap [(size_t)(gi*n_conf_ap+ai)] = r;
                    gal_rap2[(size_t)(gi*n_conf_ap+ai)] = r*r;
                    rm = std::max(rm, r);
                }
                gal_rmax_c[(size_t)gi] = rm;
                cf_xlo=std::min(cf_xlo,gal_xc[(size_t)gi]-rm); cf_xhi=std::max(cf_xhi,gal_xc[(size_t)gi]+rm);
                cf_ylo=std::min(cf_ylo,gal_yc[(size_t)gi]-rm); cf_yhi=std::max(cf_yhi,gal_yc[(size_t)gi]+rm);
                cf_zlo=std::min(cf_zlo,gal_zc[(size_t)gi]-rm); cf_zhi=std::max(cf_zhi,gal_zc[(size_t)gi]+rm);
            }
            sz_conf = (size_t)(n_gal*n_conf_ap);
            th_ntot.assign((size_t)nthreads, std::vector<int64_t>(sz_conf, 0));
            th_nlr .assign((size_t)nthreads, std::vector<int64_t>(sz_conf, 0));
            th_mtot.assign((size_t)nthreads, std::vector<double> (sz_conf, 0.0));
            th_mlr .assign((size_t)nthreads, std::vector<double> (sz_conf, 0.0));

            LOG() << "[bprop] confrac: n_gal=" << n_gal
                  << "  n_conf_ap=" << n_conf_ap
                  << "  rref=" << rref_col;
            do_confrac = true;
        }
    }

    if (!do_bprop && !do_confrac) {
        LOG() << "[bprop] nothing to compute";
        return res;
    }

    // ========================================================================
    // STAR CHUNK CALLBACK
    // Returns false to signal early stop (all galaxies done, no spatial aps).
    // ========================================================================
    auto star_chunk = [&](const double* sx, const double* sy, const double* sz,
                          const float* smass, const float* sage, const float* smetal,
                          const int64_t* sid, int64_t nb) -> bool
    {
        if (!do_bprop || nb==0) return true;

        // Convert conformal birth time → age [Gyr]
        age_gyr_buf.resize((size_t)nb);
        conf_to_age_gyr_batch(sage, age_gyr_buf.data(), nb, ctbl, z_snap);

        // -------------------------------------------------------------------
        // Member matching (OMP parallel over chunk particles)
        // -------------------------------------------------------------------
#ifdef VPP_USE_OMP
#pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            auto &fc  = th_found  [(size_t)tid];
            auto &mL  = th_mb_lum [(size_t)tid];
            auto &mS  = th_mb_sfr [(size_t)tid];
            auto &mM  = th_mb_mass[(size_t)tid];
            std::fill(fc.begin(), fc.end(), 0LL);

#pragma omp for schedule(static)
            for (int64_t i=0; i<nb; ++i) {
                auto it = member_map.find(sid[(size_t)i]);
                if (it==member_map.end()) continue;
                const int64_t gi = it->second;
                if (gal_done[(size_t)gi]) continue;
                const float metal = smetal[(size_t)i]; if (metal<=0.0f) continue;
                const double age = (double)age_gyr_buf[(size_t)i];
                const double m   = (double)smass[(size_t)i] * msun;

                mM[(size_t)gi] += m;
                fc[(size_t)gi]++;

                if (n_mag_ap>0 && n_flux>0) {
                    for (int32_t fi=0; fi<n_flux; ++fi) {
                        const int bi = band_idx[(size_t)fi]; if (bi<0) continue;
                        const double ml = interp_ml(miles, bi, age, (double)metal);
                        const double L  = (ml>0.0) ? m/ml : 0.0;
                        for (int32_t ai=0; ai<n_mag_ap; ++ai) {
                            if (!mag_use_mb[(size_t)ai]) continue;
                            mL[(size_t)(gi*n_mag_ap*n_flux+ai*n_flux+fi)] += L;
                        }
                    }
                }
                for (int32_t si=0; si<n_sfr; ++si) {
                    if (!sfr_use_mb[(size_t)si]) continue;
                    if (age < vh.sfr_t[(size_t)si]) mS[(size_t)(gi*n_sfr+si)] += m;
                }
            }
        }
#else
        {
            auto &fc  = th_found  [0];
            auto &mL  = th_mb_lum [0];
            auto &mS  = th_mb_sfr [0];
            auto &mM  = th_mb_mass[0];
            std::fill(fc.begin(), fc.end(), 0LL);
            for (int64_t i=0; i<nb; ++i) {
                auto it = member_map.find(sid[(size_t)i]);
                if (it==member_map.end()) continue;
                const int64_t gi = it->second;
                if (gal_done[(size_t)gi]) continue;
                const float metal = smetal[(size_t)i]; if (metal<=0.0f) continue;
                const double age = (double)age_gyr_buf[(size_t)i];
                const double m   = (double)smass[(size_t)i] * msun;
                mM[(size_t)gi] += m;
                fc[(size_t)gi]++;
                if (n_mag_ap>0 && n_flux>0) {
                    for (int32_t fi=0; fi<n_flux; ++fi) {
                        const int bi = band_idx[(size_t)fi]; if (bi<0) continue;
                        const double ml = interp_ml(miles, bi, age, (double)metal);
                        const double L  = (ml>0.0) ? m/ml : 0.0;
                        for (int32_t ai=0; ai<n_mag_ap; ++ai) {
                            if (!mag_use_mb[(size_t)ai]) continue;
                            mL[(size_t)(gi*n_mag_ap*n_flux+ai*n_flux+fi)] += L;
                        }
                    }
                }
                for (int32_t si=0; si<n_sfr; ++si) {
                    if (!sfr_use_mb[(size_t)si]) continue;
                    if (age < vh.sfr_t[(size_t)si]) mS[(size_t)(gi*n_sfr+si)] += m;
                }
            }
        }
#endif
        // Merge found counts; update done status
        for (int64_t gi=0; gi<n_gal; ++gi) {
            if (gal_done[(size_t)gi]) continue;
            int64_t cf = 0;
            for (int t=0; t<nthreads; ++t) cf += th_found[(size_t)t][(size_t)gi];
            found_total[(size_t)gi] += cf;
            if (found_total[(size_t)gi] >= total_members[(size_t)gi]) {
                gal_done[(size_t)gi] = true;
                ++n_done;
            }
        }

        // Early stop when all galaxies done and no spatial apertures needed
        if (n_done==n_gal && !has_finite_ap) return false;

        // -------------------------------------------------------------------
        // Spatial aperture queries via per-chunk kd-tree
        // -------------------------------------------------------------------
        if (!has_finite_ap) return true;

        // Compute chunk AABB
        double cxlo=sx[0],cxhi=sx[0], cylo=sy[0],cyhi=sy[0], czlo=sz[0],czhi=sz[0];
        for (int64_t i=1; i<nb; ++i) {
            if (sx[(size_t)i]<cxlo) cxlo=sx[(size_t)i]; if (sx[(size_t)i]>cxhi) cxhi=sx[(size_t)i];
            if (sy[(size_t)i]<cylo) cylo=sy[(size_t)i]; if (sy[(size_t)i]>cyhi) cyhi=sy[(size_t)i];
            if (sz[(size_t)i]<czlo) czlo=sz[(size_t)i]; if (sz[(size_t)i]>czhi) czhi=sz[(size_t)i];
        }
        // Global AABB check — skip chunk if no galaxy's sphere overlaps
        if (cxhi<sp_xlo||cxlo>sp_xhi||cyhi<sp_ylo||cylo>sp_yhi||czhi<sp_zlo||czlo>sp_zhi)
            return true;

        // Build kd-tree on this chunk
        KDTree ctree;
        ctree.build(sx, sy, sz, nb);

        // Per-galaxy spatial queries (OMP parallel over galaxies)
#ifdef VPP_USE_OMP
#pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            auto &sL = th_sp_lum[(size_t)tid];
            auto &sS = th_sp_sfr[(size_t)tid];
            std::vector<int32_t> hits;

#pragma omp for schedule(dynamic, 1)
            for (int64_t gi=0; gi<n_gal; ++gi) {
                const double mr = gal_max_r[(size_t)gi];
                if (mr<=0.0) continue;
                if (gal_xc[(size_t)gi]+mr<cxlo||gal_xc[(size_t)gi]-mr>cxhi||
                    gal_yc[(size_t)gi]+mr<cylo||gal_yc[(size_t)gi]-mr>cyhi||
                    gal_zc[(size_t)gi]+mr<czlo||gal_zc[(size_t)gi]-mr>czhi) continue;

                hits.clear();
                ctree.query_radius(sx,sy,sz,
                                   gal_xc[(size_t)gi],gal_yc[(size_t)gi],gal_zc[(size_t)gi],
                                   mr, hits);

                for (int32_t hi : hits) {
                    const float metal = smetal[(size_t)hi]; if (metal<=0.0f) continue;
                    auto mit = member_map.find(sid[(size_t)hi]);
                    if (mit == member_map.end() || mit->second != gi) continue;
                    const double age = (double)age_gyr_buf[(size_t)hi];
                    const double m   = (double)smass[(size_t)hi] * msun;
                    const double dex = sx[(size_t)hi]-gal_xc[(size_t)gi];
                    const double dey = sy[(size_t)hi]-gal_yc[(size_t)gi];
                    const double dez = sz[(size_t)hi]-gal_zc[(size_t)gi];
                    const double d   = std::sqrt(dex*dex+dey*dey+dez*dez);

                    double L[32] = {};
                    for (int32_t fi=0; fi<n_flux; ++fi) {
                        const int bi = band_idx[(size_t)fi]; if (bi<0) continue;
                        const double ml = interp_ml(miles, bi, age, (double)metal);
                        L[(size_t)fi] = (ml>0.0) ? m/ml : 0.0;
                    }
                    for (int32_t ai=0; ai<n_mag_ap; ++ai) {
                        if (mag_use_mb[(size_t)ai]) continue;
                        if (d > vh.mag_r[(size_t)ai]*gal_rhalf[(size_t)gi]) continue;
                        const size_t base = (size_t)(gi*n_mag_ap*n_flux+ai*n_flux);
                        for (int32_t fi=0; fi<n_flux; ++fi) sL[base+(size_t)fi] += L[(size_t)fi];
                    }
                    for (int32_t si=0; si<n_sfr; ++si) {
                        if (sfr_use_mb[(size_t)si]) continue;
                        if (d > vh.sfr_r[(size_t)si]*gal_rhalf[(size_t)gi]) continue;
                        if (age < vh.sfr_t[(size_t)si]) sS[(size_t)(gi*n_sfr+si)] += m;
                    }
                }
            }
        }
#else
        {
            auto &sL = th_sp_lum[0];
            auto &sS = th_sp_sfr[0];
            std::vector<int32_t> hits;
            for (int64_t gi=0; gi<n_gal; ++gi) {
                const double mr = gal_max_r[(size_t)gi];
                if (mr<=0.0) continue;
                if (gal_xc[(size_t)gi]+mr<cxlo||gal_xc[(size_t)gi]-mr>cxhi||
                    gal_yc[(size_t)gi]+mr<cylo||gal_yc[(size_t)gi]-mr>cyhi||
                    gal_zc[(size_t)gi]+mr<czlo||gal_zc[(size_t)gi]-mr>czhi) continue;
                hits.clear();
                ctree.query_radius(sx,sy,sz,
                                   gal_xc[(size_t)gi],gal_yc[(size_t)gi],gal_zc[(size_t)gi],
                                   mr, hits);
                for (int32_t hi : hits) {
                    const float metal = smetal[(size_t)hi]; if (metal<=0.0f) continue;
                    auto mit = member_map.find(sid[(size_t)hi]);
                    if (mit == member_map.end() || mit->second != gi) continue;
                    const double age = (double)age_gyr_buf[(size_t)hi];
                    const double m   = (double)smass[(size_t)hi] * msun;
                    const double dex = sx[(size_t)hi]-gal_xc[(size_t)gi];
                    const double dey = sy[(size_t)hi]-gal_yc[(size_t)gi];
                    const double dez = sz[(size_t)hi]-gal_zc[(size_t)gi];
                    const double d   = std::sqrt(dex*dex+dey*dey+dez*dez);
                    double L[32] = {};
                    for (int32_t fi=0; fi<n_flux; ++fi) {
                        const int bi = band_idx[(size_t)fi]; if (bi<0) continue;
                        const double ml = interp_ml(miles, bi, age, (double)metal);
                        L[(size_t)fi] = (ml>0.0) ? m/ml : 0.0;
                    }
                    for (int32_t ai=0; ai<n_mag_ap; ++ai) {
                        if (mag_use_mb[(size_t)ai]) continue;
                        if (d > vh.mag_r[(size_t)ai]*gal_rhalf[(size_t)gi]) continue;
                        const size_t base = (size_t)(gi*n_mag_ap*n_flux+ai*n_flux);
                        for (int32_t fi=0; fi<n_flux; ++fi) sL[base+(size_t)fi] += L[(size_t)fi];
                    }
                    for (int32_t si=0; si<n_sfr; ++si) {
                        if (sfr_use_mb[(size_t)si]) continue;
                        if (d > vh.sfr_r[(size_t)si]*gal_rhalf[(size_t)gi]) continue;
                        if (age < vh.sfr_t[(size_t)si]) sS[(size_t)(gi*n_sfr+si)] += m;
                    }
                }
            }
        }
#endif
        return true;
    }; // end star_chunk

    // ========================================================================
    // DM CHUNK CALLBACK
    // Returns false to signal early stop (currently unused — always true).
    // ========================================================================
    auto dm_chunk = [&](const double* dx, const double* dy, const double* dz,
                        const float* dmass, const int64_t* /*did*/, int64_t nb) -> bool
    {
        if (!do_confrac || nb==0) return true;

        // Compute chunk AABB
        double cxlo=dx[0],cxhi=dx[0], cylo=dy[0],cyhi=dy[0], czlo=dz[0],czhi=dz[0];
        for (int64_t i=1; i<nb; ++i) {
            if (dx[(size_t)i]<cxlo) cxlo=dx[(size_t)i]; if (dx[(size_t)i]>cxhi) cxhi=dx[(size_t)i];
            if (dy[(size_t)i]<cylo) cylo=dy[(size_t)i]; if (dy[(size_t)i]>cyhi) cyhi=dy[(size_t)i];
            if (dz[(size_t)i]<czlo) czlo=dz[(size_t)i]; if (dz[(size_t)i]>czhi) czhi=dz[(size_t)i];
        }
        // Global confrac AABB check — skip chunks with no galaxy overlap
        if (cxhi<cf_xlo||cxlo>cf_xhi||cyhi<cf_ylo||cylo>cf_yhi||czhi<cf_zlo||czlo>cf_zhi)
            return true;

        // Build kd-tree on this DM chunk
        KDTree ctree;
        ctree.build(dx, dy, dz, nb);

        // Per-galaxy confrac queries (OMP parallel over galaxies)
#ifdef VPP_USE_OMP
#pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            auto &nt = th_ntot[(size_t)tid];
            auto &nl = th_nlr [(size_t)tid];
            auto &mt = th_mtot[(size_t)tid];
            auto &ml = th_mlr [(size_t)tid];
            std::vector<int32_t> hits;

#pragma omp for schedule(dynamic, 1)
            for (int64_t gi=0; gi<n_gal; ++gi) {
                const double rmax = gal_rmax_c[(size_t)gi];
                if (rmax<=0.0) continue;
                if (gal_xc[(size_t)gi]+rmax<cxlo||gal_xc[(size_t)gi]-rmax>cxhi||
                    gal_yc[(size_t)gi]+rmax<cylo||gal_yc[(size_t)gi]-rmax>cyhi||
                    gal_zc[(size_t)gi]+rmax<czlo||gal_zc[(size_t)gi]-rmax>czhi) continue;

                hits.clear();
                ctree.query_radius(dx,dy,dz,
                                   gal_xc[(size_t)gi],gal_yc[(size_t)gi],gal_zc[(size_t)gi],
                                   rmax, hits);

                for (int32_t hi : hits) {
                    const float m = dmass[(size_t)hi];
                    const bool is_lr = (m>lr_thresh);
                    const double ex = dx[(size_t)hi]-gal_xc[(size_t)gi];
                    const double ey = dy[(size_t)hi]-gal_yc[(size_t)gi];
                    const double ez = dz[(size_t)hi]-gal_zc[(size_t)gi];
                    const double d2 = ex*ex+ey*ey+ez*ez;
                    for (int32_t ai=0; ai<n_conf_ap; ++ai) {
                        if (d2>gal_rap2[(size_t)(gi*n_conf_ap+ai)]) continue;
                        const size_t idx = (size_t)(gi*n_conf_ap+ai);
                        nt[idx]++; mt[idx]+=(double)m;
                        if (is_lr) { nl[idx]++; ml[idx]+=(double)m; }
                    }
                }
            }
        }
#else
        {
            auto &nt=th_ntot[0], &nl=th_nlr[0];
            auto &mt=th_mtot[0], &ml=th_mlr[0];
            std::vector<int32_t> hits;
            for (int64_t gi=0; gi<n_gal; ++gi) {
                const double rmax = gal_rmax_c[(size_t)gi];
                if (rmax<=0.0) continue;
                if (gal_xc[(size_t)gi]+rmax<cxlo||gal_xc[(size_t)gi]-rmax>cxhi||
                    gal_yc[(size_t)gi]+rmax<cylo||gal_yc[(size_t)gi]-rmax>cyhi||
                    gal_zc[(size_t)gi]+rmax<czlo||gal_zc[(size_t)gi]-rmax>czhi) continue;
                hits.clear();
                ctree.query_radius(dx,dy,dz,
                                   gal_xc[(size_t)gi],gal_yc[(size_t)gi],gal_zc[(size_t)gi],
                                   rmax, hits);
                for (int32_t hi : hits) {
                    const float m = dmass[(size_t)hi];
                    const bool is_lr = (m>lr_thresh);
                    const double ex = dx[(size_t)hi]-gal_xc[(size_t)gi];
                    const double ey = dy[(size_t)hi]-gal_yc[(size_t)gi];
                    const double ez = dz[(size_t)hi]-gal_zc[(size_t)gi];
                    const double d2 = ex*ex+ey*ey+ez*ez;
                    for (int32_t ai=0; ai<n_conf_ap; ++ai) {
                        if (d2>gal_rap2[(size_t)(gi*n_conf_ap+ai)]) continue;
                        const size_t idx = (size_t)(gi*n_conf_ap+ai);
                        nt[idx]++; mt[idx]+=(double)m;
                        if (is_lr) { nl[idx]++; ml[idx]+=(double)m; }
                    }
                }
            }
        }
#endif
        return true;
    }; // end dm_chunk

    // ========================================================================
    // STREAMING
    // ========================================================================
    if (vh.newramses==0) {
        LOG() << "[bprop] streaming (old Ramses binary)";
        stream_ptcl_old(vh, snap, star_chunk, dm_chunk);
    } else {
        LOG() << "[bprop] streaming stars from " << part_fn;
        stream_stars_hdf(part_fn, nthreads, star_chunk);
        if (do_confrac) {
            LOG() << "[bprop] streaming DM for confrac";
            stream_dm_hdf(part_fn, nthreads, dm_chunk);
        }
    }

    // ========================================================================
    // FINALIZE BPROP
    // ========================================================================
    if (do_bprop) {
        // Merge thread accumulators
        std::vector<double> mb_lum(szlum,0.0), mb_sfr(szsfr,0.0), mb_mass((size_t)n_gal,0.0);
        std::vector<double> sp_lum(szlum,0.0), sp_sfr(szsfr,0.0);
        for (int t=0; t<nthreads; ++t) {
            for (size_t j=0; j<szlum; ++j) { mb_lum[j]+=th_mb_lum[(size_t)t][j]; sp_lum[j]+=th_sp_lum[(size_t)t][j]; }
            for (size_t j=0; j<szsfr; ++j) { mb_sfr[j]+=th_mb_sfr[(size_t)t][j]; sp_sfr[j]+=th_sp_sfr[(size_t)t][j]; }
            for (int64_t gi=0; gi<n_gal; ++gi) mb_mass[(size_t)gi]+=th_mb_mass[(size_t)t][(size_t)gi];
        }

        const auto &cat_rh_kpc = cat.f64.at("R_HalfMass");
        for (int64_t gi=0; gi<n_gal; ++gi) {
            const double rhalf_kpc = cat_rh_kpc[(size_t)gi];

            // ABmag / SBF
            for (int32_t ai=0; ai<n_mag_ap; ++ai) {
                const bool   use_mb  = mag_use_mb[(size_t)ai];
                const double dval    = vh.mag_r[(size_t)ai];
                const double dsz_kpc = use_mb ? -1.0 : dval*rhalf_kpc;
                const size_t base    = (size_t)(gi*n_mag_ap*n_flux+ai*n_flux);
                for (int32_t fi=0; fi<n_flux; ++fi) {
                    const int bi = band_idx[(size_t)fi]; if (bi<0) continue;
                    const double sumL = use_mb ? mb_lum[base+(size_t)fi] : sp_lum[base+(size_t)fi];
                    if (sumL<=0.0) continue;
                    const double msun_b = msun_band(bi);
                    res.abmag[base+(size_t)fi] = msun_b - 2.5*std::log10(sumL);
                    if (dsz_kpc>0.0) {
                        const double area = (dsz_kpc*1e3)*(dsz_kpc*1e3);
                        const double sbf  = sumL/area;
                        res.sbf[base+(size_t)fi] = (sbf>0.0) ? msun_b+21.572-2.5*std::log10(sbf) : NaN;
                    }
                }
            }

            // SFR
            for (int32_t si=0; si<n_sfr; ++si) {
                const bool   use_mb = sfr_use_mb[(size_t)si];
                const double tw     = vh.sfr_t[(size_t)si];
                const double summ   = use_mb ? mb_sfr[(size_t)(gi*n_sfr+si)] : sp_sfr[(size_t)(gi*n_sfr+si)];
                res.sfr[(size_t)(gi*n_sfr+si)] = (tw>0.0) ? summ/(tw*1e9) : 0.0;
            }

            // Clump classification
            if (do_clump) {
                const double mt = (*cat_mtot_ptr)[(size_t)gi]*msun;
                const double s0 = res.sfr[(size_t)(gi*n_sfr+0)];
                res.isclump[(size_t)gi] = (s0>mt/(vh.sfr_t[0]*1e9)*vh.pp_clump_mfrac) ? 1 : -1;
            }
        }
        LOG() << "[bprop] photometry done  (n_done=" << n_done << "/" << n_gal << ")";
    }

    // ========================================================================
    // FINALIZE CONFRAC
    // ========================================================================
    if (do_confrac) {
        for (int64_t gi=0; gi<n_gal; ++gi) {
            for (int32_t ai=0; ai<n_conf_ap; ++ai) {
                const size_t idx = (size_t)(gi*n_conf_ap+ai);
                int64_t nt=0,nl=0; double mt=0.0,ml=0.0;
                for (int t=0; t<nthreads; ++t) {
                    nt+=th_ntot[(size_t)t][idx]; nl+=th_nlr[(size_t)t][idx];
                    mt+=th_mtot[(size_t)t][idx]; ml+=th_mlr[(size_t)t][idx];
                }
                res.confrac_n[idx] = (nt>0) ? (double)nl/(double)nt : 0.0;
                res.confrac_m[idx] = (mt>0.0) ? ml/mt : 0.0;
            }
        }
        LOG() << "[bprop] confrac done";
    }

    LOG() << "[bprop] all done";
    return res;
}

} // namespace Pipeline
