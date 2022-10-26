#include <cstdint>

using std::size_t;

#include <highfive/H5Exception.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Object.hpp>
#include <highfive/H5PropertyList.hpp>

using HighFive::Chunking;
using HighFive::DataSetCreateProps;
using HighFive::DataSpace;
using HighFive::File;

namespace {

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

}  // namespace


extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0) return 0;

    // Open a file
    File file("/dev/shm/sync-write.h5", File::Overwrite);

    // Create DataSet
    constexpr int height = 1;
    const auto width = size;
    constexpr int chunk_height = 1;

    auto props = DataSetCreateProps::Default();
    props.add(Chunking{chunk_height, width});
    props.add(Jpegls{});

    // Compress and write data
    auto dset = file.createDataSet<uint8_t>("/dset1", DataSpace{height, width}, props);
    dset.write_raw(data);

    // Read and decompress data
    std::vector<uint8_t> decoded(height * width);
    dset.read((uint8_t**) decoded.data());

    const bool is_equal = std::equal(decoded.begin(), decoded.end(), data);

    return (is_equal ? 0 : -1);  // Values other than 0 and -1 are reserved for future use.
}