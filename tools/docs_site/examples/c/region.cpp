#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#include "radveil.h"

// RAII ownership for the opaque volume handle.
struct VolumeFree { void operator()(RadveilVolume *v) const { radveil_volume_free(v); } };
using Volume = std::unique_ptr<RadveilVolume, VolumeFree>;

static void check(int32_t rc) {
    if (rc != 0) throw std::runtime_error(radveil_last_error());
}

int main() {
    Volume vol{radveil_volume_load("t1.nii.gz")};
    if (!vol) throw std::runtime_error(radveil_last_error());

    uint64_t dims[3];
    radveil_volume_shape(vol.get(), dims);
    std::vector<uint8_t> mask(dims[0] * dims[1] * dims[2]);  // C order, k fastest

    RadveilDefaceParams params;
    radveil_deface_params_init(&params);  // every field "auto" ...
    params.skin_depth_mm = 4.0;           // ... then override what you need

    RadveilAutoReport rep;
    check(radveil_anonymization_mask(vol.get(), "adaptive", "face", &params, nullptr,
                                     /*brain=*/0, mask.data(), &rep));
    std::printf("%llu voxels would be rewritten; the volume is untouched\n",
                (unsigned long long)rep.base.voxels_changed);
}
