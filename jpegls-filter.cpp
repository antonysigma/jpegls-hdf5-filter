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

span<uint8_t>
encode(span<uint8_t> raw, const subchunk_config_t c) {
    std::vector<byte_array_t> local_out(c.subchunks);

    // For each sub-chunk of raw data, determine the byte range, image width and height.
    // Then, compress data.
#pragma omp parallel for schedule(guided)
    for (size_t block = 0; block < c.subchunks; block++) {
        const size_t width = c.length;

        // Let's say chunk height is 27, not divisible by 24. We have the
        // remainder of 3. We distribute the remainder by adding additional
        // single row to the first 3 subchunks.
        const size_t padded_height = c.lblocks + 1;
        const size_t height = (block < c.remainder) ? padded_height : c.lblocks;

        // Now, for the first 3 subchunks, the offset is computed by padded
        // heights and the block id. The rest has an additional offsets.
        const size_t offset =
            c.typesize * width *
            ((block < c.remainder) ? height * block
                                   : (padded_height * c.remainder + height * (block - c.remainder)));

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
        out_buf = {raw.data, compressed_size};
    } else {
        out_buf = {static_cast<uint8_t*>(realloc(raw.data, compressed_size)), compressed_size};
    }

    span<uint32_t> header{reinterpret_cast<uint32_t*>(out_buf.data), c.subchunks};

#pragma omp parallel for schedule(guided)
    for (size_t block = 0; block < c.subchunks; block++) {
        const auto offset = std::accumulate(
            local_out.begin(), local_out.begin() + block, size_t(c.header_size),
            [](const auto& a, const auto& b) -> size_t { return a + b.size(); });

        const auto& local_buf = local_out[block];
        // Write header
        header[block] = local_buf.size();

        // Write payload
        std::copy(local_buf.begin(), local_buf.end(), out_buf.begin() + offset);
    }

    return out_buf;
}

#ifdef H5JPEGLS_USE_ASYNC
std::array<tf::Task, 3>
encodeAsync(span<const uint8_t> raw, const subchunk_config_t c, tf::Taskflow& taskflow,
            encode_ctx_t& encoded) {
    constexpr size_t zero = 0;
    constexpr size_t one = 1;
    const auto n_subchunks = c.subchunks;

    auto allocate_task = taskflow.emplace([&, n_subchunks]() {
        // Allocate buffers of the subchunks.
        encoded = encode_cache_t{n_subchunks};
    });

    // For each sub-chunk of raw data, determine the byte range, image width and height.
    // Then, compress data.
    auto scatter_task =
        taskflow.for_each_index(zero, n_subchunks, one, [&, c, raw](const size_t block) {
            const size_t width = c.length;
            const size_t height =
                (c.remainder != 0 && block == c.subchunks) ? c.remainder : c.lblocks;
            const size_t offset = c.typesize * width * height * block;
            const image_buffer_t<const uint8_t> input{
                raw.subspan(offset, width * height * c.typesize), c.typesize, width, height, 1};

            auto& local_out = std::get<encode_cache_t>(encoded).local_out.at(block);
            local_out = encodeSubchunk(input);
        });

    // Compute the total compressed size in bytes. We will shrink wrap the
    // compressed subchunks into one contiguous data layout.
    auto shrink_task = taskflow.emplace([&, c]() {
        auto& compressed_size = std::get<encode_cache_t>(encoded).compressed_size;
        const auto& local_out = std::get<encode_cache_t>(encoded).local_out;

        compressed_size =
            std::accumulate(local_out.begin(), local_out.end(), c.header_size,
                            [](const auto& a, const auto& b) -> size_t { return a + b.size(); });
    });

    auto gather_task = taskflow.emplace([&, c]() {
        const size_t compressed_size = std::get<encode_cache_t>(encoded).compressed_size;
        const auto& local_out = std::get<encode_cache_t>(encoded).local_out;

        byte_array_t encoded_buf(compressed_size);

        span<uint32_t> header{reinterpret_cast<uint32_t*>(encoded_buf.data()), c.subchunks};

        for (size_t block = 0; block < c.subchunks; block++) {
            const auto offset = std::accumulate(
                local_out.begin(), local_out.begin() + block, c.header_size,
                [](const auto& a, const auto& b) -> size_t { return a + b.size(); });

            const auto local_buf = std::move(local_out.at(block));

            // Write header
            header[block] = local_buf.size();

            // Write payload
            std::copy(local_buf.begin(), local_buf.end(), encoded_buf.begin() + offset);
        }

        // move the aggregated data to the output buffer
        encoded = std::move(encoded_buf);
    });

    // Now, label the tasks for debugging purpose.
    allocate_task.name("allocate");
    scatter_task.name("compress");
    shrink_task.name("shrink");
    gather_task.name("gather");

    // Schecule the tasks serially.
    taskflow.linearize({allocate_task, scatter_task, shrink_task, gather_task});

    // Return the tasks for a more fine grain task scheduling, e.g. concurrency limit.
    return {allocate_task, scatter_task, gather_task};
}
#endif
}  // namespace jpegls
