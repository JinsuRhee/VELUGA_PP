#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/*
 * run_bh_potential — Barnes-Hut tree gravitational potential solver.
 *
 * All positions in kpc (relative to any fixed reference frame).
 * All masses in Msun.
 * Output pot[i] in (km/s)^2  (negative: potential well).
 *
 * G used: 4.302e-6 (km/s)^2 kpc/Msun.
 *
 * n_thread: number of OMP threads to use (0 → use OMP max_threads).
 * bsize   : BH tree leaf bucket size (default 512, reduced automatically for small N).
 */
void run_bh_potential(int n_ptcl,
                      const double *x, const double *y, const double *z,
                      const double *mass,
                      double *pot,
                      int n_thread,
                      int bsize);

#ifdef __cplusplus
}
#endif
