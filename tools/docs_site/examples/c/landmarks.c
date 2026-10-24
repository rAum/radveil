#include <math.h>
#include <stdio.h>
#include "radveil.h"

/* Your own detector: 68 landmarks (standard 68-point order), world RAS+ mm.
 * Here they are read from "points.txt" (one "x y z" per line, "nan nan nan"
 * for a missing point). To lift 2D detections yourself, render with
 * radveil_render: its pos output maps every pixel to a world position. */
int main(void) {
    double pts[68 * 3];
    FILE *f = fopen("points.txt", "r");
    if (!f) { perror("points.txt"); return 1; }
    for (int i = 0; i < 68 * 3; i++)
        if (fscanf(f, "%lf", &pts[i]) != 1) pts[i] = NAN;
    fclose(f);

    RadveilVolume *vol = radveil_volume_load("t1.nii.gz");
    if (!vol) { fprintf(stderr, "%s\n", radveil_last_error()); return 1; }
    RadveilReport rep;
    if (radveil_deface(vol, pts, "adaptive", NULL, NULL, &rep) != 0) {
        fprintf(stderr, "%s\n", radveil_last_error());
        radveil_volume_free(vol);
        return 1;
    }
    printf("%llu voxels changed (no verification: check the result)\n",
           (unsigned long long)rep.voxels_changed);
    radveil_volume_save(vol, "t1_defaced.nii.gz", 1);
    radveil_volume_free(vol);
    return 0;
}
