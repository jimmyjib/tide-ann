// Owner labels: a membership value names the server rank that owns a node, so
// every label must be below the server count K. The server (--num-servers) and
// the coordinator (its --servers list) check every membership file at load and
// refuse to start (or to reset) otherwise. Labels written later come from
// validated sources (a migration plan checked against K, an insert routed to a
// checked label). Every routing site below the load uses owner_label_checked,
// which stops the process with a message instead of folding a corrupt label
// (label % K) onto another server: a folded label searches the wrong server and
// later fails the controller's K-owner checks (frame capture, merge, STOP).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace tide {

[[noreturn]] inline void owner_label_fatal(uint32_t label, uint32_t servers, const char* where) {
    std::fprintf(stderr, "FATAL: owner label %u is not below the server count %u (%s)\n", label, servers,
                 where ? where : "?");
    std::fflush(stderr);
    std::abort();
}

// `label` when it is below `servers`; otherwise the process stops (never folded).
inline uint32_t owner_label_checked(uint32_t label, uint32_t servers, const char* where) {
    if (label >= servers) owner_label_fatal(label, servers, where);
    return label;
}

// True when every label is below `servers` (and servers > 0); otherwise false
// with the first offending node in `error`.
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
