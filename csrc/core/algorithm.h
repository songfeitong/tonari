#pragma once

#include "errors.h"

#include <string_view>


namespace neighbor_search {

enum class Algorithm {
    Auto,
    BruteForce,
    CellList,
};


inline Algorithm parse_algorithm(std::string_view value) {
    if (value == "auto") {
        return Algorithm::Auto;
    }
    if (value == "brute_force") {
        return Algorithm::BruteForce;
    }
    if (value == "cell_list") {
        return Algorithm::CellList;
    }
    throw InputError(
        "algorithm must be 'auto', 'brute_force', or 'cell_list'");
}

}  // namespace neighbor_search
