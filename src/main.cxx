// main.cxx — VELUGA Post-Processor (VELUGA_PP)
// MPI+OpenMP wrapper, mirrors veluga_pp.pro pipeline

#ifdef VPP_USE_MPI
#include <mpi.h>
#endif
#ifdef VPP_USE_OMP
#include <omp.h>
#endif

#include "global/allvar.h"
#include "pipeline/pipeline.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

//-----
static int g_rank()
{
    int r = 0;
#ifdef VPP_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
#endif
    return r;
}

static void print_logo(int myrank)
{
    if (myrank != 0) return;
    std::cout << "\n";
    std::cout << "  ██╗   ██╗███████╗██╗      ██╗   ██╗ ██████╗  █████╗     ██████╗ ██████╗ \n";
    std::cout << "  ██║   ██║██╔════╝██║      ██║   ██║██╔════╝ ██╔══██╗    ██╔══██╗██╔══██╗\n";
    std::cout << "  ██║   ██║█████╗  ██║      ██║   ██║██║  ███╗███████║    ██████╔╝██████╔╝\n";
    std::cout << "  ╚██╗ ██╔╝██╔══╝  ██║      ██║   ██║██║   ██║██╔══██║    ██╔═══╝ ██╔═══╝ \n";
    std::cout << "   ╚████╔╝ ███████╗███████╗ ╚██████╔╝╚██████╔╝██║  ██║    ██║     ██║     \n";
    std::cout << "    ╚═══╝  ╚══════╝╚══════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝     \n";
    std::cout << "\n";
    std::cout << "  VELUGA Post-Processor  (VELUGA_PP)\n";
    std::cout << "  Written by Jinsu Rhee (jinsu.rhee@gmail.com)\n";
    std::cout << "  Last commit: " << GIT_COMMIT_HASH << "\n";
    std::cout << "\n";
}

static void print_config(const vpp_set::Settings &vh, int myrank)
{
    if (myrank != 0) return;
    LOG() << "---- Configuration ----";
    LOG() << "  dir_raw          = " << vh.dir_raw;
    LOG() << "  dir_catalog      = " << vh.dir_catalog;
    LOG() << "  horg             = " << vh.horg;
    LOG() << "  snap range       = " << vh.snapi << " -> " << vh.snapf << " (step " << vh.snapd << ")";
    LOG() << "  newramses        = " << vh.newramses
          << (vh.newramses == 0 ? " (old Fortran binary)" : " (new HDF5)");
    LOG() << "  column_list      = " << vh.column_list.size() << " columns"
          << (vh.column_list.empty() ? " (all)" : "");
    LOG() << "-----------------------";
}

//-----
int main(int argc, char **argv)
{
#ifdef VPP_USE_MPI
    MPI_Init(&argc, &argv);
#endif
    int myrank = g_rank();

    print_logo(myrank);

    if (argc < 2) {
        if (myrank == 0)
            std::cerr << "Usage: veluga_pp <config_file>\n";
#ifdef VPP_USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    if (myrank == 0) {
        LOG() << "Program starts";
        LOG() << "  Start time: " << vpp_log::timestamp_full();
        LOG() << "  Last commit: " << GIT_COMMIT_HASH;
    }

    //-----
    // Load configuration
    //-----
    vpp_set::Settings vh;
    if (!g_load_config(argv[1], vh)) {
        if (myrank == 0)
            LOG() << "Failed to load config: " << argv[1];
#ifdef VPP_USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    print_config(vh, myrank);

#ifdef VPP_USE_OMP
    if (myrank == 0)
        LOG() << "  OMP threads: " << omp_get_max_threads();
#endif

#ifdef VPP_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    //-----
    // Main snapshot loop
    //-----
    if (myrank == 0)
        LOG() << "Entering main snapshot loop";

    for (int32_t snap = vh.snapi; snap <= vh.snapf; snap += vh.snapd) {

        auto t0 = std::chrono::steady_clock::now();

        if (myrank == 0)
            LOG() << "===== Snapshot " << snap << " =====";

        //-- Skip if catalog file is absent
        {
            const std::string probe = Pipeline::catalog_probe(vh, snap);
            struct stat _st{};
            if (::stat(probe.c_str(), &_st) != 0) {
                if (myrank == 0)
                    LOG() << "  catalog not found, skipping: " << probe;
                continue;
            }
        }

        //-- Phase 1: load_catalog (rv_RawCatalog)
        auto cat = Pipeline::load_catalog(vh, snap);
#ifdef VPP_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        //-- Phase 2: read_ptclid (rv_ReadID)
        auto ptcl = Pipeline::read_ptclid(vh, snap);
#ifdef VPP_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        //-- Phase 3: member_match (rv_PTMatch)
        Pipeline::member_match(vh, snap);
#ifdef VPP_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        //-- Phase 4: compute_bulk (rv_BProp)
        Pipeline::compute_bulk(vh, snap, cat, ptcl);
#ifdef VPP_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        //-- Phase 5: save (rv_save)
        Pipeline::save(vh, snap);
#ifdef VPP_USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        if (myrank == 0) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            LOG() << "  snap " << snap << " done in "
                  << std::fixed << std::setprecision(2) << dt << "s"
                  << "  (n=" << cat.n << ")";
        }
    }

    if (myrank == 0)
        LOG() << "Program end";

#ifdef VPP_USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
