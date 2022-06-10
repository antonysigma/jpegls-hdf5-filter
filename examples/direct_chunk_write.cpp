#include <cstdint>
#include <iostream>
#include <vector>

#include <highfive/H5Exception.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Object.hpp>
#include <highfive/H5PropertyList.hpp>

#if H5_VERSION_LE(1, 10, 2)
#include <hdf5_hl.h>
#endif

#include "jpegls-filter.h"

using HighFive::Chunking;
using HighFive::DataSetCreateProps;
using HighFive::DataSpace;
using HighFive::File;

namespace {

class TaskflowTimeProfiler {
    std::shared_ptr<tf::ChromeObserver> observer;

   public:
    /** Create the default observer */
    TaskflowTimeProfiler(tf::Executor* executor)
        : observer((executor) ? executor->make_observer<tf::ChromeObserver>() : nullptr) {}

    /** When this object goes out of scope, dump the time profiling data
     * to the system `/tmp` folder.
     */
    ~TaskflowTimeProfiler() {
        if (observer == nullptr) {
            return;
        }

        std::cout << "Exporting time profiling data...\n";

        std::ofstream tracing_file("/tmp/tracing.json", std::ofstream::trunc);

        // dump the execution timeline to json (view at chrome://tracing)
        observer->dump(tracing_file);

        tracing_file.close();
    }
};

class Jpegls {
   public:
    explicit Jpegls() = default;

   private:
    const std::array<uint32_t, 3> filter_param{0, 0, 0};
    friend HighFive::DataSetCreateProps;
    friend HighFive::GroupCreateProps;

    inline void apply(const hid_t hid) const {
        const auto status =
            H5Pset_filter(hid, 32012, H5Z_FLAG_MANDATORY, filter_param.size(), filter_param.data());

        if (status < 0) {
            HighFive::HDF5ErrMapper::ToException<HighFive::PropertyException>(
                "Error enabling Jpeg-LS filter");
        }
    }
};

template <class Dataset, typename T>
herr_t
writeChunk(Dataset& dset, const std::array<hsize_t, 2> offset, std::vector<T>&& src_buffer) {
    const auto dset_id = dset.getId();
    constexpr auto filter_mask = 0;

#if H5_VERSION_LE(1, 10, 2)
    return H5DOwrite_chunk(dset_id, H5P_DEFAULT, filter_mask, offset.data(), src_buffer.size(),
                           src_buffer.data());
#else
    return H5Dwrite_chunk(dset_id, H5P_DEFAULT, filter_mask, offset.data(), src_buffer.size(),
                          src_buffer.data());
#endif
}

}  // namespace

int
main() {
    // Open a file
    File file("testing.h5", File::Overwrite);

    // Create DataSet
    constexpr int height = 512;
    constexpr int width = 512;
    constexpr int chunk_height = 64;

    auto props = DataSetCreateProps::Default();
    props.add(Chunking{chunk_height, width});
    props.add(Jpegls{});

    auto dset = file.createDataSet<uint16_t>("/dset1", DataSpace{height, width}, props);

    {
        const std::vector<uint16_t> ones(height * width, 1);

        // Write ones
        dset.write_raw(ones.data());
    }

    // Allocate memory
    tf::Taskflow taskflow;
    jpegls::encode_ctx_t encoded;
    const jpegls::subchunk_config_t config(width, chunk_height, sizeof(uint16_t));

    const std::vector<uint16_t> twos(chunk_height * width, 2);

    auto [A, B, C] =
        encodeAsync(jpegls::span<const uint8_t>{reinterpret_cast<const uint8_t*>(twos.data()),
                                                twos.size() * sizeof(uint16_t)},
                    config, taskflow, encoded);

    auto write_task = taskflow
                          .emplace([&]() {
                              const auto status = writeChunk(
                                  dset, {0, 0}, std::move(std::get<jpegls::byte_array_t>(encoded)));
                              if (status < 0) {
                                  std::cerr << "Error: " << status << '\n';
                              }
                          })
                          .name("Write");

    // Now, schedule the tasks
    C.precede(write_task);

    // Now execute all tasks in multi-threaded environment
    tf::Executor executor;

    // (Optional) Profile the time spent on each task
    constexpr bool export_benchmark = true;
    TaskflowTimeProfiler profiler((export_benchmark) ? &executor : nullptr);

    executor.run(taskflow).wait();

    return 0;
}