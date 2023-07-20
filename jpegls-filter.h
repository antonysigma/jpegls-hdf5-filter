#pragma once
#include <array>
#include <cstdint>
#include <vector>

#ifdef H5JPEGLS_USE_ASYNC
#include <taskflow/taskflow.hpp>
#endif

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
    int32_t typesize = 1;
    size_t nblocks = 1;
    size_t subchunks = 1;
    size_t lblocks = 1;
    size_t header_size = sizeof(uint32_t);
    size_t remainder = 0;
    size_t lossy = 0;

    constexpr subchunk_config_t(int l, size_t _nblocks, int32_t t, int _lossy = 0)
        : length(l),
          typesize(t),
          nblocks(_nblocks),
          subchunks(std::min(size_t(24), nblocks)),
          lblocks(nblocks / subchunks),
          header_size(sizeof(uint32_t) * subchunks),
          remainder(nblocks - lblocks * subchunks),
          lossy(_lossy) {}
};

/** Compress one chunk of data, defined by the HDF5 chunk shape.
 * @param[in] raw input data pointer and byte count.
 * @param[in] config sub-chunk data layout to compress in parallel.
 * @param[out] encoded encoded data.
 */
span<uint8_t>
encode(span<uint8_t> buffer, const subchunk_config_t config);

#ifdef H5JPEGLS_USE_ASYNC

using byte_array_t = std::vector<uint8_t>;

struct encode_cache_t {
    size_t compressed_size;
    std::vector<uint32_t> block_size;
    std::vector<byte_array_t> local_out;

    encode_cache_t() = default;

    encode_cache_t(size_t N) : block_size(N), local_out(N) {}
};

using encode_ctx_t = std::variant<encode_cache_t, byte_array_t>;

/** Encode the chunk asychronously */
std::array<tf::Task, 3> encodeAsync(span<const uint8_t> raw, const subchunk_config_t config,
                                    tf::Taskflow& taskflow, encode_ctx_t& encoded);
#endif
}  // namespace jpegls
