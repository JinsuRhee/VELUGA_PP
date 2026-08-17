// miles.cxx — MILES SPS table loader + bilinear ML interpolation
// Tables: sdss_ch_iPp0.00.MAG and NUV_ch_iPp0.00.MAG
// Row order in file: (met_0, age_0..49), (met_1, age_0..49), ... (n_met groups × n_age rows)
// 2D table stored as ml[age_idx * n_met + met_idx]

#include "pipeline/miles.h"
#include "global/allvar.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace Pipeline {

int band_index(const std::string &name)
{
    std::string lo = name;
    for (auto &c : lo) c = (char)std::tolower((unsigned char)c);
    if (lo == "u")   return BAND_U;
    if (lo == "g")   return BAND_G;
    if (lo == "r")   return BAND_R;
    if (lo == "i")   return BAND_I;
    if (lo == "z")   return BAND_Z;
    if (lo == "nuv") return BAND_NUV;
    return -1;
}

// -----------------------------------------------------------------------
// Internal row struct for parsing
// -----------------------------------------------------------------------

struct RawRow {
    double Z;    // log10([M/H]) column from file
    double age;  // Gyr
    double ml;   // mass-to-light ratio
};

static std::vector<RawRow> parse_table(const std::string &path, int ml_col)
{
    std::ifstream in(path);
    if (!in) {
        LOG() << "[miles] ERROR: cannot open " << path;
        return {};
    }

    std::string line;
    std::getline(in, line); // skip header

    std::vector<RawRow> rows;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string model;
        double mu, Z, age, mass;
        iss >> model >> mu >> Z >> age >> mass;

        // Read up to ml_col remaining columns
        double v = 0.0;
        for (int c = 0; c <= ml_col; ++c) {
            if (!(iss >> v)) { v = -99.0; break; }
        }
        rows.push_back({Z, age, v});
    }
    return rows;
}

// -----------------------------------------------------------------------
// Build 2D ML table [age_idx * n_met + met_idx] from flat row data
// Rows assumed to be sorted by (metallicity group, then age within group)
// -----------------------------------------------------------------------

// Nearest index in a sorted vector (within tolerance)
static int nearest_idx(const std::vector<double> &v, double x)
{
    if (v.empty()) return -1;
    auto it = std::lower_bound(v.begin(), v.end(), x);
    if (it == v.end()) return (int)(v.size() - 1);
    if (it == v.begin()) return 0;
    auto prev = std::prev(it);
    return (std::abs(*it - x) < std::abs(*prev - x))
           ? (int)(it - v.begin()) : (int)(prev - v.begin());
}

static void fill_ml(const std::vector<RawRow> &rows,
                    const std::vector<double> &ages,
                    const std::vector<double> &mets,
                    double sun_met,
                    std::vector<double> &out)
{
    const int na = (int)ages.size();
    const int nm = (int)mets.size();
    out.assign((size_t)(na * nm), 0.0);

    for (size_t r = 0; r < rows.size(); ++r) {
        double met_abs = std::pow(10.0, rows[r].Z) * sun_met;
        int ai = nearest_idx(ages, rows[r].age);
        int mi = nearest_idx(mets, met_abs);
        if (ai < 0 || ai >= na || mi < 0 || mi >= nm) continue;
        out[(size_t)(ai * nm + mi)] = rows[r].ml;
    }

    // Replace negative ML with min positive in each metallicity column
    for (int mi = 0; mi < nm; ++mi) {
        double min_pos = 1e300;
        for (int ai = 0; ai < na; ++ai) {
            double v = out[(size_t)(ai * nm + mi)];
            if (v > 0.0 && v < min_pos) min_pos = v;
        }
        if (min_pos == 1e300) min_pos = 1e-5; // fallback
        for (int ai = 0; ai < na; ++ai) {
            double &v = out[(size_t)(ai * nm + mi)];
            if (v <= 0.0) v = min_pos;
        }
    }
}

// -----------------------------------------------------------------------
// load_miles
// -----------------------------------------------------------------------

MilesML load_miles(double sun_met)
{
    const std::string dir_table = VPP_TABLE_DIR;
    const std::string sdss_path = dir_table + "/sdss_ch_iPp0.00.MAG";
    const std::string nuv_path  = dir_table + "/NUV_ch_iPp0.00.MAG";

    // Load all 5 SDSS ML columns (indices 5..9 in the remaining stream after mass)
    std::vector<RawRow> sdss_u = parse_table(sdss_path, 5);
    std::vector<RawRow> sdss_g = parse_table(sdss_path, 6);
    std::vector<RawRow> sdss_r = parse_table(sdss_path, 7);
    std::vector<RawRow> sdss_i = parse_table(sdss_path, 8);
    std::vector<RawRow> sdss_z = parse_table(sdss_path, 9);
    // NUV: model mu Z age mass NUV ML_NUV → ML_NUV is column 1 after mass
    std::vector<RawRow> nuv    = parse_table(nuv_path,  1);

    if (sdss_u.empty()) {
        LOG() << "[miles] ERROR: failed to load SDSS table from " << sdss_path;
        return {};
    }

    // Discover unique ages and metallicities (use sdss_u as reference)
    std::set<double> age_set, met_set;
    for (auto &r : sdss_u) {
        age_set.insert(r.age);
        met_set.insert(std::round(std::pow(10.0, r.Z) * sun_met * 1e8) / 1e8);
    }
    std::vector<double> ages(age_set.begin(), age_set.end());
    std::vector<double> mets(met_set.begin(), met_set.end());
    std::sort(ages.begin(), ages.end());
    std::sort(mets.begin(), mets.end());

    MilesML tbl;
    tbl.age   = ages;
    tbl.met   = mets;
    tbl.n_age = (int32_t)ages.size();
    tbl.n_met = (int32_t)mets.size();

    fill_ml(sdss_u, ages, mets, sun_met, tbl.ml[BAND_U]);
    fill_ml(sdss_g, ages, mets, sun_met, tbl.ml[BAND_G]);
    fill_ml(sdss_r, ages, mets, sun_met, tbl.ml[BAND_R]);
    fill_ml(sdss_i, ages, mets, sun_met, tbl.ml[BAND_I]);
    fill_ml(sdss_z, ages, mets, sun_met, tbl.ml[BAND_Z]);
    fill_ml(nuv,    ages, mets, sun_met, tbl.ml[BAND_NUV]);

    LOG() << "[miles] loaded: n_age=" << tbl.n_age << "  n_met=" << tbl.n_met
          << "  age=[" << ages.front() << "," << ages.back() << "]"
          << "  met=[" << mets.front() << "," << mets.back() << "]";
    return tbl;
}

// -----------------------------------------------------------------------
// interp_ml — bilinear interpolation
// -----------------------------------------------------------------------

double interp_ml(const MilesML &tbl, int band_idx, double age_gyr, double met_abs)
{
    if (tbl.n_age < 2 || tbl.n_met < 2) return 0.0;
    if (band_idx < 0 || band_idx >= N_BANDS) return 0.0;

    const auto &ml = tbl.ml[band_idx];

    // Clamp to table range
    age_gyr = std::max(age_gyr, tbl.age.front());
    age_gyr = std::min(age_gyr, tbl.age.back());
    met_abs = std::max(met_abs, tbl.met.front());
    met_abs = std::min(met_abs, tbl.met.back());

    // Find lower bracket for age
    int ai = (int)(std::lower_bound(tbl.age.begin(), tbl.age.end(), age_gyr)
                   - tbl.age.begin());
    if (ai == tbl.n_age) --ai;
    if (ai > 0 && (ai == tbl.n_age - 1 || age_gyr < tbl.age[(size_t)ai])) --ai;

    // Find lower bracket for metallicity
    int mi = (int)(std::lower_bound(tbl.met.begin(), tbl.met.end(), met_abs)
                   - tbl.met.begin());
    if (mi == tbl.n_met) --mi;
    if (mi > 0 && (mi == tbl.n_met - 1 || met_abs < tbl.met[(size_t)mi])) --mi;

    // Ensure upper indices in range
    int ai1 = std::min(ai + 1, tbl.n_age - 1);
    int mi1 = std::min(mi + 1, tbl.n_met - 1);

    const int nm = tbl.n_met;
    double z00 = ml[(size_t)(ai  * nm + mi )];
    double z01 = ml[(size_t)(ai  * nm + mi1)];
    double z10 = ml[(size_t)(ai1 * nm + mi )];
    double z11 = ml[(size_t)(ai1 * nm + mi1)];

    // Interpolate along age axis
    double da = tbl.age[(size_t)ai1] - tbl.age[(size_t)ai];
    double ta = (da > 0.0) ? (age_gyr - tbl.age[(size_t)ai]) / da : 0.0;
    double zz0 = z00 + ta * (z10 - z00);
    double zz1 = z01 + ta * (z11 - z01);

    // Interpolate along metallicity axis
    double dm = tbl.met[(size_t)mi1] - tbl.met[(size_t)mi];
    double tm = (dm > 0.0) ? (met_abs - tbl.met[(size_t)mi]) / dm : 0.0;
    return zz0 + tm * (zz1 - zz0);
}

} // namespace Pipeline
