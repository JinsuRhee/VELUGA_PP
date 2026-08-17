#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef VPP_USE_MPI
#include <mpi.h>
#endif

//-----
// LOG macro
//-----
#define LOG() vpp_log::Line(vpp_log::basename_cstr(__FILE__), __LINE__)

//-----
// Settings
//-----
namespace vpp_set
{
    struct Settings
    {
        // Paths
        std::string dir_raw     = "";   // raw RAMSES simulation directory
        std::string dir_catalog = "";   // VELOCIraptor catalog root

        // Snapshot range
        int32_t snapi = -1;
        int32_t snapf = -1;
        int32_t snapd =  1;

        // halo ('h') or galaxy ('g')
        char    horg  = 'g';

        // 0 = old Ramses Fortran binary, 1 = new Ramses HDF5
        int32_t newramses = 0;

        // Columns to read from the VR properties file (empty = all)
        std::vector<std::string> column_list;

        // Flux bands for magnitude/SBF computation (e.g. "NUV","u","g","r","i","z")
        std::vector<std::string> flux_list = {};

        // SFR: per-entry aperture (× R_HalfMass, negative = all particles) and time window [Gyr]
        std::vector<double> sfr_r = {};
        std::vector<double> sfr_t = {};

        // Magnitude/SBF apertures (× R_HalfMass, negative = all particles)
        std::vector<double> mag_r = {};

        // Clump correction: galaxy is a clump if SFR[0] > mass_star * pp_clump_mfrac / SFR_T[0]
        // Set to 0 to disable clump classification
        double pp_clump_mfrac = 0.0;

        // Contamination fraction apertures (× R_HalfMass for g, × R_200crit for h)
        std::vector<double> conf_r = {};

        // Effective DM particle count per dimension (for HR DM mass formula).
        // HR DM mass [code] = (omega_m - omega_b) / (omega_m * neff^3)
        // Set to 0 to auto-detect from min(dm.mass).
        int32_t neff = 0;

        // Gas mass apertures (× R_HalfMass for g, × R_200crit for h)
        std::vector<double> gas_r = {};

        // Gravitational aperture for potential computation + pressure (× same rref as above).
        // Cells + particles within this radius are used for the BH-tree potential.
        // Set to 0 to skip potential computation and energy classification.
        double grav_r = 0.0;

        // Mean molecular weight for temperature computation from P/rho.
        // 1.2195 ≈ 1 / (X/1 + Y/4) for standard H/He mix (fully ionized: ~0.6, neutral: ~1.4).
        double mu_mean = 1.2195;

        // Skip flags — set to 1 in config to skip the corresponding computation
        int32_t skip_bprop    = 0;  // skip galaxy photometry / SFR / isclump
        int32_t skip_confrac  = 0;  // skip contamination fraction
        int32_t skip_gasprop  = 0;  // skip gas mass + pressure computation
    };
}

//-----
// Logging
//-----
namespace vpp_log
{
    inline const char *basename_cstr(const char *path)
    {
        const char *base = path;
        for (const char *p = path; *p; ++p)
            if (*p == '/' || *p == '\\') base = p + 1;
        return base;
    }

    inline std::string timestamp_full()
    {
        using namespace std::chrono;
        const auto now = system_clock::now();
        const auto tt  = system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y:%m:%d:%H:%M:%S");
        return oss.str();
    }

    inline std::chrono::steady_clock::time_point &start_tp()
    {
        static std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
        return t;
    }

    inline std::string elapsed_sec_str()
    {
        using namespace std::chrono;
        const auto elapsed = duration_cast<seconds>(steady_clock::now() - start_tp()).count();
        std::ostringstream oss;
        oss << '[' << std::setw(7) << elapsed << "s]";
        return oss.str();
    }

    class Line
    {
    public:
        Line(const char *code, int line) : code_(code), line_(line) {}
        ~Line()
        {
            std::ostream &out = std::cout;
            out << elapsed_sec_str()
                << " [" << std::setw(16) << code_ << "] "
                << "[" << std::setw(4) << line_ << "]  "
                << ss_.str() << "\n";
        }
        template <typename T>
        Line &operator<<(const T &v) { ss_ << v; return *this; }
    private:
        const char *code_;
        int          line_;
        std::ostringstream ss_;
    };
}

//-----
// MPI helpers
//-----
#ifdef VPP_USE_MPI
namespace mpi_type
{
    template <typename T> MPI_Datatype type();
    template <> inline MPI_Datatype type<int32_t>()  { return MPI_INT32_T;  }
    template <> inline MPI_Datatype type<int64_t>()  { return MPI_INT64_T;  }
    template <> inline MPI_Datatype type<uint32_t>() { return MPI_UINT32_T; }
    template <> inline MPI_Datatype type<uint64_t>() { return MPI_UINT64_T; }
    template <> inline MPI_Datatype type<float>()    { return MPI_FLOAT;    }
    template <> inline MPI_Datatype type<double>()   { return MPI_DOUBLE;   }
}
#endif

//-----
// Global function declarations
//-----
bool g_load_config(const std::string &path, vpp_set::Settings &vh);
