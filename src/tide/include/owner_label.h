// Owner labels must be < K, and owner_label_checked aborts instead of folding (label % K).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace tide {

[[noreturn]] inline void owner_label_fatal(uint32_t label, uint32_t servers, const char* where) {
    std::fprintf(stderr, "Error: owner label %u is not below the server count %u (%s)\n", label, servers,
                 where ? where : "?");
    std::fflush(stderr);
    std::abort();
}

// `label` if below `servers`, else the process stops.
inline uint32_t owner_label_checked(uint32_t label, uint32_t servers, const char* where) {
    if (label >= servers) owner_label_fatal(label, servers, where);
    return label;
}

// False with the first bad node in `error` unless every label is below `servers`.
inline bool owner_labels_valid(const std::vector<uint32_t>& labels, uint32_t servers, std::string* error) {
    if (servers == 0) {
        if (error) *error = "the server count is zero";
        return false;
    }
    for (size_t v = 0; v < labels.size(); ++v) {
        if (labels[v] >= servers) {
            if (error) {
                *error = "node " + std::to_string(v) + " has owner label " + std::to_string(labels[v]) +
                         ", not below the server count " + std::to_string(servers);
            }
            return false;
        }
    }
    if (error) error->clear();
    return true;
}

}  // namespace tide
