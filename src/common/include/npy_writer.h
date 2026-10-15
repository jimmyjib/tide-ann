// Writers for .npy files and DiskANN adjacency files.
// NPY v1.0: magic, version, header padded to 64 bytes, data.
#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace ann {

inline void write_npy_float(const std::string& path, const std::vector<float>& data,
                            uint32_t rows, uint32_t cols) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);

    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    file.write(magic, 6);

    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char*>(&major), 1);
    file.write(reinterpret_cast<const char*>(&minor), 1);

    std::ostringstream header_stream;
    header_stream << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                  << rows << ", " << cols << "), }";
    std::string header = header_stream.str();

    size_t total_len = 10 + header.size() + 1;  // magic(6) + version(2) + header_len(2)
    size_t padding = (64 - (total_len % 64)) % 64;
    header += std::string(padding, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header.data(), header.size());

    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

// checked_close: throw on any write or close failure.
inline void write_npy_uint32(const std::string& path, const std::vector<uint32_t>& data,
                             bool checked_close = false) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);
    if (checked_close) file.exceptions(std::ios::badbit | std::ios::failbit);

    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    file.write(magic, 6);

    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char*>(&major), 1);
    file.write(reinterpret_cast<const char*>(&minor), 1);

    std::ostringstream header_stream;
    header_stream << "{'descr': '<u4', 'fortran_order': False, 'shape': ("
                  << data.size() << ",), }";
    std::string header = header_stream.str();

    size_t total_len = 10 + header.size() + 1;
    size_t padding = (64 - (total_len % 64)) % 64;
    header += std::string(padding, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header.data(), header.size());

    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
    if (checked_close) { file.flush(); file.close(); }
}

inline void write_npy_uint32_2d(const std::string& path, const std::vector<uint32_t>& data,
                                uint32_t rows, uint32_t cols) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);

    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    file.write(magic, 6);

    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char*>(&major), 1);
    file.write(reinterpret_cast<const char*>(&minor), 1);

    std::ostringstream header_stream;
    header_stream << "{'descr': '<u4', 'fortran_order': False, 'shape': ("
                  << rows << ", " << cols << "), }";
    std::string header = header_stream.str();

    size_t total_len = 10 + header.size() + 1;
    size_t padding = (64 - (total_len % 64)) % 64;
    header += std::string(padding, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header.data(), header.size());

    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
}

// DiskANN adjacency file: 24-byte header (u64 file size, u32 max degree, u32 start node, u64 reserved),
// then per node a u32 degree and the neighbor ids.
inline void write_diskann_adj(const std::string& path,
                              const std::vector<std::vector<uint32_t>>& adj,
                              uint32_t start_node) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);

    uint64_t body_size = 0;
    uint32_t max_degree = 0;
    for (const auto& neighbors : adj) {
        body_size += 4 + neighbors.size() * 4;
        max_degree = std::max(max_degree, static_cast<uint32_t>(neighbors.size()));
    }
    uint64_t file_size = 24 + body_size;

    uint64_t reserved = 0;
    file.write(reinterpret_cast<const char*>(&file_size), 8);
    file.write(reinterpret_cast<const char*>(&max_degree), 4);
    file.write(reinterpret_cast<const char*>(&start_node), 4);
    file.write(reinterpret_cast<const char*>(&reserved), 8);

    for (const auto& neighbors : adj) {
        uint32_t degree = static_cast<uint32_t>(neighbors.size());
        file.write(reinterpret_cast<const char*>(&degree), 4);
        if (degree > 0) {
            file.write(reinterpret_cast<const char*>(neighbors.data()), degree * 4);
        }
    }
}

}  // namespace ann
