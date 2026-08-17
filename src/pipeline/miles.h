#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace Pipeline {

// Band index constants (matches order in MilesML arrays)
enum BandIdx { BAND_U=0, BAND_G=1, BAND_R=2, BAND_I=3, BAND_Z=4, BAND_NUV=5, N_BANDS=6 };

// Absolute magnitude of the Sun in each band (for mag formula)
inline double msun_band(int b)
{
    constexpr double msun[N_BANDS] = {6.55, 5.12, 4.68, 4.57, 4.54, 10.18};
    return msun[b];
}

// Return BandIdx from band name string (case-insensitive u/g/r/i/z/NUV)
// Returns -1 if not found
int band_index(const std::string &name);

// MILES SPS mass-to-light table
// Layout: ml_X[age_idx * n_met + met_idx]
struct MilesML {
    std::vector<double> age;   // n_age unique ages [Gyr], ascending
    std::vector<double> met;   // n_met unique metallicities (absolute fraction), ascending
    int32_t n_age = 0;
    int32_t n_met = 0;
    std::vector<double> ml[N_BANDS]; // ML ratios per band
};

// Load MILES tables from VPP_TABLE_DIR (reads sdss_ch_iPp0.00.MAG and NUV_ch_iPp0.00.MAG)
// sun_met = 0.02 (solar metallicity in absolute fraction)
MilesML load_miles(double sun_met = 0.02);

// Bilinear interpolation of ML at (age_gyr, met_abs)
// Clamps to table boundaries; returns 0 if table is empty
double interp_ml(const MilesML &tbl, int band_idx, double age_gyr, double met_abs);

} // namespace Pipeline
