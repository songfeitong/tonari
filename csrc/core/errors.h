#pragma once

#include <stdexcept>
#include <string>


namespace neighbor_search {

class InputError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};


class SearchError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};


inline void require_input(bool condition, const char* message) {
    if (!condition) {
        throw InputError(message);
    }
}


inline void require_search(bool condition, const char* message) {
    if (!condition) {
        throw SearchError(message);
    }
}

}  // namespace neighbor_search
