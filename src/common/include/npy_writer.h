#pragma once

/**
 * NPY File Writer Utilities.
 *
 * Functions for writing numpy .npy files and DiskANN adjacency lists.
 * Used by the coordinator (membership and result files) and the tests.
 */

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace ann {

/**
 * Write a 2D float32 array to NPY file.
 *
 * @param path Path to the .npy file
 * @param data Vector of floats (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 */
inline void write_npy_float(const std::string& path, const std::vector<float>& data,
                            uint32_t rows, uint32_t cols) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);

    // NPY format: magic + version + header_len + header + data
    // Magic number
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    file.write(magic, 6);

    // Version 1.0
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char*>(&major), 1);
    file.write(reinterpret_cast<const char*>(&minor), 1);

    // Build header string
    std::ostringstream header_stream;
    header_stream << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                  << rows << ", " << cols << "), }";
    std::string header = header_stream.str();

    // Pad header to multiple of 64 bytes (including 10-byte preamble)
    size_t total_len = 10 + header.size() + 1;  // 10 = magic(6) + version(2) + header_len(2)
    size_t padding = (64 - (total_len % 64)) % 64;
    header += std::string(padding, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header.data(), header.size());

    // Write data
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

/**
 * Write a 1D uint32 array to NPY file.
 *
 * @param path Path to the .npy file
 * @param data Vector of uint32_t
 */
inline void write_npy_uint32(const std::string& path, const std::vector<uint32_t>& data,
                             bool checked_close = false) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);
    if (checked_close) file.exceptions(std::ios::badbit | std::ios::failbit);

    // Magic number
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    file.write(magic, 6);

    // Version 1.0
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char*>(&major), 1);
    file.write(reinterpret_cast<const char*>(&minor), 1);

    // Build header string (uint32 - compatible with read_npy_uint32)
    std::ostringstream header_stream;
    header_stream << "{'descr': '<u4', 'fortran_order': False, 'shape': ("
                  << data.size() << ",), }";
    std::string header = header_stream.str();

    // Pad header
    size_t total_len = 10 + header.size() + 1;
    size_t padding = (64 - (total_len % 64)) % 64;
    header += std::string(padding, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header.data(), header.size());

    // Write uint32 directly
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
    if (checked_close) { file.flush(); file.close(); }
}

/**
 * Write a 2D uint32 array to NPY file.
 *
 * @param path Path to the .npy file
 * @param data Vector of uint32_t (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 */
inline void write_npy_uint32_2d(const std::string& path, const std::vector<uint32_t>& data,
                                uint32_t rows, uint32_t cols) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);

    // Magic number
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    file.write(magic, 6);

    // Version 1.0
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char*>(&major), 1);
    file.write(reinterpret_cast<const char*>(&minor), 1);

    // Build header string (uint32 - compatible with read_npy_uint32)
    std::ostringstream header_stream;
    header_stream << "{'descr': '<u4', 'fortran_order': False, 'shape': ("
                  << rows << ", " << cols << "), }";
    std::string header = header_stream.str();

    // Pad header
    size_t total_len = 10 + header.size() + 1;
    size_t padding = (64 - (total_len % 64)) % 64;
    header += std::string(padding, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(header.data(), header.size());

    // Write uint32 directly
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
}

/**
 * Write DiskANN format adjacency list.
 *
 * Format:
 * - Header (24 bytes):
 *   - uint64_t file_size
 *   - uint32_t max_degree
 *   - uint32_t start_node
 *   - uint64_t reserved (padding)
 * - Body (for each node):
 *   - uint32_t degree
 *   - uint32_t[degree] neighbors
 *
 * @param path Path to the adj file
 * @param adj Adjacency list (adj[local_idx] = [global neighbors])
 * @param start_node Start node index for search
 */
inline void write_diskann_adj(const std::string& path,
                              const std::vector<std::vector<uint32_t>>& adj,
                              uint32_t start_node) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create: " + path);

    // Calculate file size and max degree
    uint64_t body_size = 0;
    uint32_t max_degree = 0;
    for (const auto& neighbors : adj) {
        body_size += 4 + neighbors.size() * 4;  // degree + neighbors
        max_degree = std::max(max_degree, static_cast<uint32_t>(neighbors.size()));
    }
    uint64_t file_size = 24 + body_size;  // header + body

    // Write header
    uint64_t reserved = 0;
    file.write(reinterpret_cast<const char*>(&file_size), 8);
    file.write(reinterpret_cast<const char*>(&max_degree), 4);
    file.write(reinterpret_cast<const char*>(&start_node), 4);
    file.write(reinterpret_cast<const char*>(&reserved), 8);

    // Write body
    for (const auto& neighbors : adj) {
        uint32_t degree = static_cast<uint32_t>(neighbors.size());
        file.write(reinterpret_cast<const char*>(&degree), 4);
        if (degree > 0) {
            file.write(reinterpret_cast<const char*>(neighbors.data()), degree * 4);
        }
    }
}

}  // namespace ann
