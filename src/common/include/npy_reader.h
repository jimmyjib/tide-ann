// Readers for numpy .npy files. Unless noted, a reader throws std::runtime_error on a dtype, order or
// shape it does not accept.
#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace ann {

// 1-D or 2-D float32 .npy, row-major. The header is not checked.
inline std::vector<float> read_npy_float(const std::string& path, uint32_t& rows, uint32_t& cols) {
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
    if (comma == std::string::npos) {
        rows = std::stoul(shape_str);
        cols = 1;
    } else {
        rows = std::stoul(shape_str.substr(0, comma));
        std::string cols_str = shape_str.substr(comma + 1);
        while (!cols_str.empty() && cols_str[0] == ' ') cols_str = cols_str.substr(1);
        cols = cols_str.empty() ? 1 : std::stoul(cols_str);
    }

    std::vector<float> data(static_cast<size_t>(rows) * cols);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return data;
}

// 2-D float32 .npy in C order.
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

// 1-D int64 (or int32) .npy, truncated to uint32. The header is not checked.
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

    bool is_int64 = header.find("'<i8'") != std::string::npos;

    std::vector<uint32_t> result(n);
    if (is_int64) {
        std::vector<int64_t> temp(n);
        file.read(reinterpret_cast<char*>(temp.data()), n * sizeof(int64_t));
        for (uint32_t i = 0; i < n; i++) {
            result[i] = static_cast<uint32_t>(temp[i]);
        }
    } else {
        std::vector<int32_t> temp(n);
        file.read(reinterpret_cast<char*>(temp.data()), n * sizeof(int32_t));
        for (uint32_t i = 0; i < n; i++) {
            result[i] = static_cast<uint32_t>(temp[i]);
        }
    }
    return result;
}

namespace detail {

// Parsed .npy header. The stream is left at the first data byte.
struct NpyHeader {
    std::string descr;
    bool fortran_order = false;
    std::vector<uint64_t> shape;
};

// Throws std::runtime_error naming `path` for anything it cannot parse.
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

// Throws if the file is shorter than `bytes`.
inline void read_npy_data_checked(std::ifstream& file, const std::string& path, char* out, size_t bytes) {
    file.read(out, static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(file.gcount()) != bytes) throw std::runtime_error(path + ": shorter than its shape");
}

}  // namespace detail

// Ground truth: 2-D [queries, k] int32, uint32 or int64 ids in C order.
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

// 1-D int32 or uint32 in C order (membership.npy, local_to_global.npy).
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

// One partition's adjacency rows, with global neighbor ids.
struct PartitionGraph {
    std::vector<std::vector<uint32_t>> adj;  // adj[local_idx] = global neighbor IDs
    std::vector<uint32_t> local_to_global;   // local_idx -> global_id
    std::vector<uint32_t> global_to_local;   // global_id -> local_idx (UINT32_MAX if not local)
    uint32_t n_local;
    uint32_t start_node;  // local start node
};

// Reads partition_dir/local_to_global.npy and partition_dir/adj (DiskANN graph layout).
inline PartitionGraph read_partition_graph(const std::string& partition_dir, uint32_t n_total) {
    PartitionGraph pg;

    uint32_t n_local;
    pg.local_to_global = read_npy_uint32(partition_dir + "/local_to_global.npy", n_local);
    pg.n_local = n_local;

    pg.global_to_local.resize(n_total, UINT32_MAX);
    for (uint32_t local_idx = 0; local_idx < n_local; local_idx++) {
        uint32_t global_id = pg.local_to_global[local_idx];
        if (global_id < n_total) {
            pg.global_to_local[global_id] = local_idx;
        }
    }

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
    }

    return pg;
}

}  // namespace ann
