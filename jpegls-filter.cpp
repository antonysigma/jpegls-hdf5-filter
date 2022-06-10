#include "jpegls-filter.h"

#include <numeric>
#include <iostream>

#include "charls/charls.h"

using byte_array_t = std::vector<uint8_t>;

namespace {

template <typename T>
struct image_buffer_t {
    jpegls::span<T> buffer;

    size_t typesize = 1;
    /** Pixel width. */
    size_t width = 0;

    /** Image height, i.e. number of pixel width. */
    size_t height = 0;

    /** Number of interleaved samples in a pixel. */
    uint32_t channels = 1;
};

/** Given one subchunk of data, compress it and return the encoded data. */
template <typename T>
byte_array_t
encodeSubchunk(const image_buffer_t<T> raw) {
    const auto reserved_size = raw.buffer.size_bytes();
    byte_array_t encoded(reserved_size + 8192);

    auto params = [&]() -> const JlsParameters {
        auto params = JlsParameters();
        params.width = raw.width;
        params.height = raw.height;
        params.bitsPerSample = raw.typesize * 8;
        params.components = raw.channels;
        return params;
    }();

    size_t csize;
    char err_msg[256];
    const CharlsApiResultType ret = JpegLsEncode(
        encoded.data(), encoded.size(), &csize,
        raw.buffer.begin(),
        raw.buffer.size_bytes(), &params, err_msg);
    if (ret != CharlsApiResultType::OK) {
        std::cerr << "JPEG-LS error: " << err_msg << '\n';
    }

    encoded.resize(csize);

    return encoded;
}

}

namespace jpegls {

int
encode(span<uint8_t> raw, const subchunk_config_t c) {
    std::vector<uint32_t> block_size(c.subchunks);
    std::vector<byte_array_t> local_out(c.subchunks);

    // For each sub-chunk of raw data, determine the byte range, image width and height.
    // Then, compress data.
#pragma omp parallel for schedule(guided)
    for (size_t block = 0; block < c.subchunks; block++) {
        const size_t width = c.length;
        const size_t height = (c.remainder != 0 && block == c.subchunks) ? c.remainder : c.lblocks;
        const size_t offset = c.typesize * width * height * block;

        const image_buffer_t<const uint8_t> input{raw.subspan(offset, width * height * c.typesize),
                                                  c.typesize, width, height, 1};

        local_out[block] = encodeSubchunk(input);
    }

    // Compute the total compressed size in bytes.
    const auto compressed_size =
        std::accumulate(local_out.begin(), local_out.end(), c.header_size,
                        [](const auto& a, const auto& b) -> size_t { return a + b.size(); });

    // Reallocate the raw buffer, if the new size is larger than original size.
    span<uint8_t> out_buf;

    if (compressed_size <= raw.size) {
        out_buf = raw;
    } else {
        out_buf = {static_cast<uint8_t*>(realloc(raw.data, compressed_size)), compressed_size};
    }

    span<uint32_t> header{reinterpret_cast<uint32_t*>(out_buf.data), c.subchunks};

#pragma omp parallel for schedule(guided)
    for (size_t block = 0; block < c.subchunks; block++) {
        const auto offset = std::accumulate(
            local_out.begin(), local_out.begin() + block, c.header_size,
            [](const auto& a, const auto& b) -> size_t { return a + b.size(); });

        const auto& local_buf = local_out[block];

        // Write header
        header[block] = local_buf.size();

        // Write payload
        std::copy(local_buf.begin(), local_buf.end(), raw.begin() + offset);
    }

    return compressed_size;
}
}