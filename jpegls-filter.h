#pragma once
#include <cstdint>
#include <vector>

namespace jpegls {

using std::size_t;

/** Lightweight implementation of std::span<uint8_t>. */
template<typename T>
struct span {
    T* data = nullptr;
    size_t size = 0;

    constexpr T* begin() const {
        return data;
    }

    constexpr T* end() const {
        return data + size;
    }

    constexpr T& operator[](const size_t i) const {
        return data[i];
    }

    constexpr size_t size_bytes() const {
        return size * sizeof(T);
    }

    constexpr span<T> subspan(size_t offset, size_t expected_length) const {
        if(offset > size) {
            return {};
        }

        if (offset + expected_length > size) {
            return {data + offset, size - offset};
        }

        return {data + offset, expected_length};
    }

    constexpr operator span<const uint8_t>() const {
        return {reinterpret_cast<const uint8_t*>(data), size * sizeof(uint8_t)};
    }
};

struct subchunk_config_t {
    size_t length = 1;
    size_t typesize = 1;
    size_t subchunks = 1;
    size_t lblocks = 1;
    size_t header_size = sizeof(uint32_t);
    size_t remainder = 0;

    constexpr subchunk_config_t(int l, size_t nblocks, size_t t)
        : length(l),
          typesize(t),
          subchunks(std::min(size_t(24), nblocks)),
          lblocks(nblocks / subchunks),
          header_size(sizeof(uint32_t) * subchunks),
          remainder(nblocks - lblocks * subchunks) {}
};
/** Compress one chunk of data, defined by the HDF5 chunk shape.
 * @param[in] raw input data pointer and byte count.
 * @param[in] config sub-chunk data layout to compress in parallel.
 * @param[out] encoded encoded data.
 */
span<uint8_t>
encode(span<uint8_t> buffer, const subchunk_config_t config);

}