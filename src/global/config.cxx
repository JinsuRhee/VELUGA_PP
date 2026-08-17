#include "global/allvar.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

namespace
{
    inline std::string trim(std::string s)
    {
        auto not_space = [](int ch){ return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return s;
    }

    inline std::string tolower_copy(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return (char)std::tolower(c); });
        return s;
    }

    // Strip surrounding quotes (single or double) and whitespace
    inline std::string strip_quotes(std::string s)
    {
        s = trim(s);
        if (s.size() >= 2) {
            char a = s.front(), b = s.back();
            if ((a == '"' && b == '"') || (a == '\'' && b == '\''))
                s = s.substr(1, s.size() - 2);
        }
        return s;
    }

    template <typename T>
    bool parse_num(const std::string &s, T &out)
    {
        std::istringstream iss(trim(s));
        iss >> out;
        return (bool)iss && iss.eof();
    }
}

bool g_load_config(const std::string &path, vpp_set::Settings &vh)
{
    std::ifstream in(path);
    if (!in) {
        LOG() << "config open failed: " << path;
        return false;
    }

    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;

        // Strip comments (#, ;, //)
        auto cut = std::min({ line.find('#'), line.find(';'), line.find("//"),
                              std::string::npos });
        if (cut != std::string::npos)
            line = line.substr(0, cut);
        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) {
            std::cerr << "config: ignore line " << lineno << " (no '=')\n";
            continue;
        }

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        auto lkey = tolower_copy(key);

        // Paths
        if      (lkey == "dir_raw")     vh.dir_raw     = strip_quotes(val);
        else if (lkey == "dir_catalog") vh.dir_catalog = strip_quotes(val);

        // Snapshot range
        else if (lkey == "snapi") parse_num(val, vh.snapi);
        else if (lkey == "snapf") parse_num(val, vh.snapf);
        else if (lkey == "snapd") parse_num(val, vh.snapd);

        // horg
        else if (lkey == "horg") {
            auto v = strip_quotes(val);
            if (!v.empty()) vh.horg = (char)std::tolower((unsigned char)v[0]);
        }

        // RAMSES format: 0=old Fortran binary, 1=new HDF5
        else if (lkey == "newramses") parse_num(val, vh.newramses);

        // Columns to extract from VR catalog (comma-separated)
        else if (lkey == "column_list") {
            std::istringstream css(strip_quotes(val));
            std::string item;
            while (std::getline(css, item, ',')) {
                item = trim(item);
                if (!item.empty()) vh.column_list.push_back(item);
            }
        }

        // Flux band list (comma-separated band names)
        else if (lkey == "flux_list") {
            std::istringstream css(strip_quotes(val));
            std::string item;
            while (std::getline(css, item, ',')) {
                item = trim(item);
                if (!item.empty()) vh.flux_list.push_back(item);
            }
        }

        // SFR and magnitude aperture/timewindow lists (comma-separated doubles)
        else if (lkey == "sfr_r" || lkey == "sfr_t" || lkey == "mag_r") {
            std::vector<double> *vec = (lkey == "sfr_r") ? &vh.sfr_r :
                                       (lkey == "sfr_t") ? &vh.sfr_t : &vh.mag_r;
            std::istringstream css(strip_quotes(val));
            std::string item;
            while (std::getline(css, item, ',')) {
                double v;
                if (parse_num(trim(item), v)) vec->push_back(v);
            }
        }

        else if (lkey == "pp_clump_mfrac") parse_num(val, vh.pp_clump_mfrac);
        else if (lkey == "conf_r") {
            std::istringstream css(strip_quotes(val));
            std::string item;
            while (std::getline(css, item, ',')) {
                double v;
                if (parse_num(trim(item), v)) vh.conf_r.push_back(v);
            }
        }
        else if (lkey == "neff") {
            double v; if (parse_num(val, v)) vh.neff = (int32_t)v;
        }

        else if (lkey == "gas_r") {
            std::istringstream css(strip_quotes(val));
            std::string item;
            while (std::getline(css, item, ',')) {
                double v;
                if (parse_num(trim(item), v)) vh.gas_r.push_back(v);
            }
        }
        else if (lkey == "grav_r")     parse_num(val, vh.grav_r);
        else if (lkey == "mu_mean")    parse_num(val, vh.mu_mean);
        else if (lkey == "skip_bprop")   { double v; if (parse_num(val, v)) vh.skip_bprop   = (int32_t)v; }
        else if (lkey == "skip_confrac") { double v; if (parse_num(val, v)) vh.skip_confrac = (int32_t)v; }
        else if (lkey == "skip_gasprop") { double v; if (parse_num(val, v)) vh.skip_gasprop = (int32_t)v; }

        else {
            std::cerr << "config: unknown key '" << key << "' at line " << lineno << "\n";
        }
    }

    return true;
}
