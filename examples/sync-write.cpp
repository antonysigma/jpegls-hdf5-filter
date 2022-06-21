#include <cstdint>
#include <iostream>
#include <vector>

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

int
main() {
    // Open a file
    File file("sync-write.h5", File::Overwrite);

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

    return 0;
}