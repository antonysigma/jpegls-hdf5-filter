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
    static constexpr std::array<uint32_t, 3> filter_param{0, 0, 0};
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

#pragma pack(1)
struct chunk_shape_t {
    uint8_t image_height;
    uint8_t image_width;
    uint8_t chunk_height;
    uint8_t chunk_width;
};
#pragma pack(pop)

}  // namespace


extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size != sizeof(chunk_shape_t)) return 0;

    // Parse parameters
    const auto [h, w, ch, cw] = *reinterpret_cast<const chunk_shape_t *>(data);

    const uint32_t height = 32u * h + 32;
    const uint32_t width = 32u * w + 32;
    const uint32_t chunk_height = 4u * ch + 4;
    const uint32_t chunk_width = 4u * cw + 4;

    //std::cout << '(' << int(height) << ',' << int(width) << ") (" << int(chunk_height) << ','
    //          << int(chunk_width) << ")\n";

    if (height < chunk_height || width < chunk_width) return 0;

    // Open a file
    File file("/dev/shm/sync-write.h5", File::Overwrite);

    // Create DataSet
    auto props = DataSetCreateProps::Default();
    props.add(Chunking{chunk_height, chunk_width});
    props.add(Jpegls{});

    // Compress and write data
    auto dset = file.createDataSet<uint8_t>("/dset1", DataSpace{height, width}, props);
    const std::vector<uint8_t> ones(height * width, 1);
    dset.write_raw(ones.data());

    // Read and decompress data
    std::vector<uint8_t> decoded(height * width);
    dset.read(decoded.data());

    const bool is_equal = std::equal(decoded.begin(), decoded.end(), ones.begin());

    return (is_equal ? 0 : -1);  // Values other than 0 and -1 are reserved for future use.
}