#include <stdio.h>
#include "radveil.h"

int main(void) {
    RadveilVolume *vol = radveil_volume_load("t1.nii.gz");
    if (!vol) { fprintf(stderr, "load: %s\n", radveil_last_error()); return 1; }

    RadveilAutoReport rep;
    if (radveil_deface_auto(vol, "adaptive", "face", NULL, NULL,
                            /*verify=*/1, /*brain=*/RADVEIL_BRAIN_CHECK, NULL, &rep) != 0) {
        fprintf(stderr, "refused: %s\n", radveil_last_error());  /* volume untouched */
        radveil_volume_free(vol);
        return 2;
    }
    printf("%llu voxels changed, %d faces left, %lld brain voxels touched\n",
           (unsigned long long)rep.base.voxels_changed, rep.residual_detections,
           (long long)rep.brain_voxels_modified);
    if (radveil_volume_save(vol, "t1_defaced.nii.gz", 1) != 0)
        fprintf(stderr, "save: %s\n", radveil_last_error());
    radveil_volume_free(vol);
    return 0;
}
