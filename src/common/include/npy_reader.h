#pragma once

/**
 * NPY File Reader Utilities.
 *
 * Common functions for reading numpy .npy files.
 * Used by the server, the coordinator and the PQ trainer.
 */

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace ann {

/**
 * Read a 2D float32 array from NPY file.
 *
 * @param path Path to the .npy file
 * @param rows Output: number of rows
 * @param cols Output: number of columns
 * @return Vector of floats (row-major order)
 */
inline std::vector<float> read_npy_float(const std::string& path, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    // Read magic number and version
    char magic[6];
    file.read(magic, 6);
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    // Read and parse header
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    // Parse shape from header
    auto shape_pos = header.find("'shape': (");
    size_t start = shape_pos + 10;
    size_t end = header.find(")", start);
    std::string shape_str = header.substr(start, end - start);

    size_t comma = shape_str.find(",");
    if (comma == std::string::npos) {
        rows = std::stoul(shape_str);
        cols = 1;
    } else {
        rows = std::stoul(shape_str.substr(0, comma));
        std::string cols_str = shape_str.substr(comma + 1);
        // Trim leading spaces
        while (!cols_str.empty() && cols_str[0] == ' ') cols_str = cols_str.substr(1);
        cols = cols_str.empty() ? 1 : std::stoul(cols_str);
    }

    // Read data
    std::vector<float> data(static_cast<size_t>(rows) * cols);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return data;
}

/**
 * Read a 2D little-endian float32 C-order NPY matrix and refuse anything else:
 * another dtype, Fortran order, a 1D array, or a file shorter than its shape.
 */
inline std::vector<float> read_npy_float_matrix_checked(const std::string& path, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    char magic[6];
    file.read(magic, 6);
    if (!file || std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error(path + ": not an NPY file");
    uint8_t major = 0, minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        file.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    if (!file) throw std::runtime_error(path + ": truncated NPY header");
    if (header.find("'descr': '<f4'") == std::string::npos)
        throw std::runtime_error(path + ": dtype is not little-endian float32");
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": not C order");
    const auto shape_pos = header.find("'shape': (");
    if (shape_pos == std::string::npos) throw std::runtime_error(path + ": NPY shape missing");
    const size_t start = shape_pos + 10;
    const std::string shape = header.substr(start, header.find(')', start) - start);
    const size_t comma = shape.find(',');
    std::string second = comma == std::string::npos ? "" : shape.substr(comma + 1);
    while (!second.empty() && second[0] == ' ') second = second.substr(1);
    if (comma == std::string::npos || second.empty()) throw std::runtime_error(path + ": not a 2D matrix");
    rows = static_cast<uint32_t>(std::stoul(shape.substr(0, comma)));
    cols = static_cast<uint32_t>(std::stoul(second));
    std::vector<float> data(static_cast<size_t>(rows) * cols);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (static_cast<size_t>(file.gcount()) != data.size() * sizeof(float))
        throw std::runtime_error(path + ": shorter than its shape");
    return data;
}

/**
 * Read a 1D or 2D int32 array from NPY file.
 *
 * @param path Path to the .npy file
 * @param rows Output: number of rows
 * @param cols Output: number of columns
 * @return Vector of uint32_t (row-major order)
 */
inline std::vector<uint32_t> read_npy_int32(const std::string& path, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    char magic[6];
    file.read(magic, 6);
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    auto shape_pos = header.find("'shape': (");
    size_t start = shape_pos + 10;
    size_t end = header.find(")", start);
    std::string shape_str = header.substr(start, end - start);

    size_t comma = shape_str.find(",");
    rows = std::stoul(shape_str.substr(0, comma));
    std::string cols_str = shape_str.substr(comma + 1);
    while (!cols_str.empty() && cols_str[0] == ' ') cols_str = cols_str.substr(1);
    cols = cols_str.empty() ? 1 : std::stoul(cols_str);

    std::vector<int32_t> temp(rows * cols);
    file.read(reinterpret_cast<char*>(temp.data()), temp.size() * sizeof(int32_t));

    std::vector<uint32_t> result(rows * cols);
    for (size_t i = 0; i < temp.size(); i++) {
        result[i] = static_cast<uint32_t>(temp[i]);
    }
    return result;
}

/**
 * Read a 1D int64 array from NPY file as uint32.
 *
 * @param path Path to the .npy file
 * @param n Output: number of elements
 * @return Vector of uint32_t
 */
inline std::vector<uint32_t> read_npy_int64_as_uint32(const std::string& path, uint32_t& n) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    char magic[6];
    file.read(magic, 6);
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    auto shape_pos = header.find("'shape': (");
    size_t start = shape_pos + 10;
    size_t end = header.find(")", start);
    std::string shape_str = header.substr(start, end - start);

    size_t comma = shape_str.find(",");
    n = std::stoul(comma != std::string::npos ? shape_str.substr(0, comma) : shape_str);

    // Check dtype
    bool is_int64 = header.find("'<i8'") != std::string::npos;

    std::vector<uint32_t> result(n);
    if (is_int64) {
        std::vector<int64_t> temp(n);
        file.read(reinterpret_cast<char*>(temp.data()), n * sizeof(int64_t));
        for (uint32_t i = 0; i < n; i++) {
            result[i] = static_cast<uint32_t>(temp[i]);
        }
    } else {
        // Try int32
        std::vector<int32_t> temp(n);
        file.read(reinterpret_cast<char*>(temp.data()), n * sizeof(int32_t));
        for (uint32_t i = 0; i < n; i++) {
            result[i] = static_cast<uint32_t>(temp[i]);
        }
    }
    return result;
}

namespace detail {

// The checked header of an NPY file: the dtype string, the memory order and the
// shape. The reader is left at the first data byte.
struct NpyHeader {
    std::string descr;
    bool fortran_order = false;
    std::vector<uint64_t> shape;
};

// Reads and checks the magic, the version and the header dictionary; throws
// std::runtime_error naming `path` for anything it cannot parse.
inline NpyHeader read_npy_header_checked(std::ifstream& file, const std::string& path) {
    char magic[6];
    file.read(magic, 6);
    if (!file || std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error(path + ": not an NPY file");
    uint8_t major = 0, minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        file.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2 || major == 3) {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error(path + ": unsupported NPY version " + std::to_string(major));
    }
    if (!file || header_len == 0 || header_len > (1u << 20)) throw std::runtime_error(path + ": truncated NPY header");
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    if (!file) throw std::runtime_error(path + ": truncated NPY header");
    NpyHeader h;
    const auto key = [&](const char* name) {
        const size_t at = header.find(name);
        if (at == std::string::npos) throw std::runtime_error(path + ": NPY header has no " + name);
        size_t pos = at + std::string(name).size();
        while (pos < header.size() && (header[pos] == ' ' || header[pos] == ':')) ++pos;
        return pos;
    };
    size_t pos = key("'descr'");
    if (pos >= header.size() || header[pos] != '\'') throw std::runtime_error(path + ": NPY dtype is not a string");
    const size_t close = header.find('\'', pos + 1);
    if (close == std::string::npos) throw std::runtime_error(path + ": NPY dtype is not a string");
    h.descr = header.substr(pos + 1, close - pos - 1);
    pos = key("'fortran_order'");
    if (header.compare(pos, 4, "True") == 0) h.fortran_order = true;
    else if (header.compare(pos, 5, "False") != 0) throw std::runtime_error(path + ": NPY fortran_order unreadable");
    pos = key("'shape'");
    if (pos >= header.size() || header[pos] != '(') throw std::runtime_error(path + ": NPY shape missing");
    const size_t end = header.find(')', pos);
    if (end == std::string::npos) throw std::runtime_error(path + ": NPY shape missing");
    std::string item;
    for (size_t i = pos + 1; i <= end; ++i) {
        const char c = header[i];
        if (c == ',' || c == ')') {
            if (!item.empty()) {
                if (item.size() > 19 || item.find_first_not_of("0123456789") != std::string::npos)
                    throw std::runtime_error(path + ": NPY shape unreadable");
                h.shape.push_back(std::stoull(item));
            }
            item.clear();
        } else if (c != ' ') {
            item.push_back(c);
        }
    }
    return h;
}

// Reads exactly `bytes` data bytes after the header; a shorter file is refused.
inline void read_npy_data_checked(std::ifstream& file, const std::string& path, char* out, size_t bytes) {
    file.read(out, static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(file.gcount()) != bytes) throw std::runtime_error(path + ": shorter than its shape");
}

}  // namespace detail

/**
 * Read ground truth from an NPY file: a C-order 2D [rows, cols] array of
 * little-endian int32, uint32 or int64 ids. Anything else is refused with
 * std::runtime_error: no NPY magic, another dtype, Fortran order, another rank,
 * a shape beyond uint32, or a file shorter than its shape.
 *
 * @param path Path to the .npy file
 * @param rows Output: number of queries
 * @param cols Output: number of neighbors per query
 * @return Vector of int32_t (row-major order)
 */
inline std::vector<int32_t> read_npy_groundtruth(const std::string& path, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    const detail::NpyHeader h = detail::read_npy_header_checked(file, path);
    if (h.descr != "<i4" && h.descr != "<u4" && h.descr != "<i8")
        throw std::runtime_error(path + ": ground truth dtype '" + h.descr + "' is not int32, uint32 or int64");
    if (h.fortran_order) throw std::runtime_error(path + ": not C order");
    if (h.shape.size() != 2) throw std::runtime_error(path + ": ground truth is not a 2D [queries, k] array");
    if (h.shape[0] > UINT32_MAX || h.shape[1] > UINT32_MAX || h.shape[1] == 0)
        throw std::runtime_error(path + ": ground truth shape out of range");
    rows = static_cast<uint32_t>(h.shape[0]);
    cols = static_cast<uint32_t>(h.shape[1]);
    const size_t count = static_cast<size_t>(rows) * cols;
    std::vector<int32_t> result(count);
    if (h.descr == "<i8") {
        std::vector<int64_t> temp(count);
        detail::read_npy_data_checked(file, path, reinterpret_cast<char*>(temp.data()), count * sizeof(int64_t));
        for (size_t i = 0; i < count; i++) result[i] = static_cast<int32_t>(temp[i]);
    } else {
        detail::read_npy_data_checked(file, path, reinterpret_cast<char*>(result.data()), count * sizeof(int32_t));
    }
    return result;
}

/**
 * Read a 1D little-endian int32 or uint32 C-order array from an NPY file
 * (membership.npy, local_to_global.npy, a --self-ids file). Any other dtype (an
 * int64 file included), rank or order, and a file shorter than its shape, is
 * refused with std::runtime_error instead of being read as raw uint32 words.
 *
 * @param path Path to the .npy file
 * @param n Output: number of elements
 * @return Vector of uint32_t
 */
inline std::vector<uint32_t> read_npy_uint32(const std::string& path, uint32_t& n) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    const detail::NpyHeader h = detail::read_npy_header_checked(file, path);
    if (h.descr != "<i4" && h.descr != "<u4")
        throw std::runtime_error(path + ": dtype '" + h.descr + "' is not int32 or uint32");
    if (h.fortran_order) throw std::runtime_error(path + ": not C order");
    if (h.shape.size() != 1) throw std::runtime_error(path + ": not a 1D array");
    if (h.shape[0] > UINT32_MAX) throw std::runtime_error(path + ": shape out of range");
    n = static_cast<uint32_t>(h.shape[0]);
    std::vector<uint32_t> result(n);
    detail::read_npy_data_checked(file, path, reinterpret_cast<char*>(result.data()),
                                  static_cast<size_t>(n) * sizeof(uint32_t));
    return result;
}

// ============================================================================
// Graph I/O Utilities
// ============================================================================

/**
 * Read DiskANN format graph file.
 *
 * File format:
 *   - 8 bytes: file_size
 *   - 4 bytes: max_degree
 *   - 4 bytes: start_node
 *   - 8 bytes: reserved
 *   - For each node: 4 bytes degree + degree*4 bytes neighbors
 *
 * @param path Path to the graph file
 * @param n Number of nodes
 * @return Pair of (adjacency list, start_node)
 */
inline std::pair<std::vector<std::vector<uint32_t>>, uint32_t>
read_graph(const std::string& path, uint32_t n) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open graph: " + path);

    uint64_t file_size;
    uint32_t max_degree, start_node;
    uint64_t reserved;
    file.read(reinterpret_cast<char*>(&file_size), 8);
    file.read(reinterpret_cast<char*>(&max_degree), 4);
    file.read(reinterpret_cast<char*>(&start_node), 4);
    file.read(reinterpret_cast<char*>(&reserved), 8);

    std::vector<std::vector<uint32_t>> adj(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t degree;
        file.read(reinterpret_cast<char*>(&degree), 4);
        adj[i].resize(degree);
        file.read(reinterpret_cast<char*>(adj[i].data()), degree * 4);
    }
    return {adj, start_node};
}

/**
 * Partition-specific graph data.
 * Used by the server for partition-graph mode.
 */
struct PartitionGraph {
    std::vector<std::vector<uint32_t>> adj;  // adj[local_idx] = global neighbor IDs
    std::vector<uint32_t> local_to_global;   // local_idx -> global_id
    std::vector<uint32_t> global_to_local;   // global_id -> local_idx (UINT32_MAX if not local)
    uint32_t n_local;
    uint32_t start_node;  // local start node
};

/**
 * Read partition-specific graph (adj[local_idx] = [global neighbors]).
 *
 * Expected files in partition_dir:
 *   - local_to_global.npy: mapping from local to global indices
 *   - adj: DiskANN format adjacency list
 *
 * @param partition_dir Directory containing partition files
 * @param n_total Total number of nodes in the full dataset
 * @return PartitionGraph structure
 */
inline PartitionGraph read_partition_graph(const std::string& partition_dir, uint32_t n_total) {
    PartitionGraph pg;

    // Read local_to_global mapping
    uint32_t n_local;
    pg.local_to_global = read_npy_uint32(partition_dir + "/local_to_global.npy", n_local);
    pg.n_local = n_local;

    // Build global_to_local mapping
    pg.global_to_local.resize(n_total, UINT32_MAX);
    for (uint32_t local_idx = 0; local_idx < n_local; local_idx++) {
        uint32_t global_id = pg.local_to_global[local_idx];
        if (global_id < n_total) {
            pg.global_to_local[global_id] = local_idx;
        }
    }

    // Read adjacency list (DiskANN format)
    std::string adj_path = partition_dir + "/adj";
    std::ifstream file(adj_path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open partition graph: " + adj_path);

    uint64_t file_size;
    uint32_t max_degree, start_node;
    uint64_t reserved;
    file.read(reinterpret_cast<char*>(&file_size), 8);
    file.read(reinterpret_cast<char*>(&max_degree), 4);
    file.read(reinterpret_cast<char*>(&start_node), 4);
    file.read(reinterpret_cast<char*>(&reserved), 8);

    pg.start_node = start_node;
    pg.adj.resize(n_local);

    for (uint32_t i = 0; i < n_local; i++) {
        uint32_t degree;
        file.read(reinterpret_cast<char*>(&degree), 4);
        pg.adj[i].resize(degree);
        file.read(reinterpret_cast<char*>(pg.adj[i].data()), degree * 4);
        // neighbors are already global IDs (written by tools/prepare/runtime_lib.write_adj,
        // which partition_graph.py calls)
    }

    return pg;
}

}  // namespace ann
