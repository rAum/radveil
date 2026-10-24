/* SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
 * SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "radveil.h"

int main(void) {
    printf("radveil %s\n", radveil_version());
    /* build a small phantom via from_array */
    size_t n = 64;
    float *data = calloc(n * n * n, sizeof(float));
    double c = n / 2.0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t k = 0; k < n; k++) {
                double x = i - c, y = j - c, z = k - c;
                double head = pow(x / 22, 2) + pow(y / 25, 2) + pow(z / 27, 2);
                double nose = pow(x / 5, 2) + pow((y + 28) / 8, 2) + pow((z + 4) / 6, 2);
                if (head <= 1.0 || nose <= 1.0) data[(i * n + j) * n + k] = 800.0f;
            }
    double aff[16] = {2, 0, 0, -64, 0, 2, 0, -64, 0, 0, 2, -64, 0, 0, 0, 1};
    RadveilVolume *vol = radveil_volume_from_array(n, n, n, data, aff);
    if (!vol) { fprintf(stderr, "load: %s\n", radveil_last_error()); return 1; }
    char orient[8];
    radveil_volume_orientation(vol, orient, sizeof orient);
    uint64_t dims[3];
    radveil_volume_shape(vol, dims);
    printf("shape %llu %llu %llu orient %s\n",
           (unsigned long long)dims[0], (unsigned long long)dims[1],
           (unsigned long long)dims[2], orient);

    uint32_t size = 128;
    float *pos = malloc(size * size * 3 * sizeof(float));
    uint8_t *flags = malloc(size * size);
    if (radveil_render(vol, 0.0, 0.0, size, 0.14f, NAN, NULL, pos, flags)) {
        fprintf(stderr, "render: %s\n", radveil_last_error()); return 1;
    }
    int hits = 0;
    for (uint32_t i = 0; i < size * size; i++) hits += flags[i] > 0;
    printf("hits %d\n", hits);

    /* built-in detector: this synthetic volume has no face -> refused */
    RadveilAutoReport arep;
    if (radveil_deface_auto(vol, "blur", "face", NULL, NULL, 1, 0, NULL, &arep) == 0) {
        fprintf(stderr, "deface_auto accepted a faceless volume\n"); return 1;
    }
    printf("deface_auto refused faceless volume: %s\n", radveil_last_error());
    uint8_t *region_mask = malloc(n * n * n);
    if (radveil_anonymization_mask(vol, "blur", "face", NULL, NULL, 0, region_mask, NULL) == 0) {
        fprintf(stderr, "anonymization_mask accepted a faceless volume\n"); return 1;
    }
    free(region_mask);

    double pts[68 * 3];
    for (int i = 0; i < 68 * 3; i++) pts[i] = NAN;
    for (int i = 36; i < 42; i++) { pts[i*3]=-16; pts[i*3+1]=-44; pts[i*3+2]=8; }
    for (int i = 42; i < 48; i++) { pts[i*3]= 16; pts[i*3+1]=-44; pts[i*3+2]=8; }
    pts[30*3]=0; pts[30*3+1]=-70; pts[30*3+2]=-8;
    for (int i = 48; i < 68; i++) { pts[i*3]=0; pts[i*3+1]=-46; pts[i*3+2]=-26; }
    RadveilDefaceParams dp;
    radveil_deface_params_init(&dp);  /* all auto ... */
    dp.skin_depth_mm = 4.0;           /* ... then override what you need */
    dp.options = RADVEIL_SPARE_AIR;   /* opt-in: leave background air untouched */
    RadveilReport rep;
    if (radveil_deface(vol, pts, "adaptive", &dp, NULL, &rep)) {
        fprintf(stderr, "deface: %s\n", radveil_last_error()); return 1;
    }
    printf("defaced %llu voxels, eye_dist %.1f mm, sigma %.1f mm\n",
           (unsigned long long)rep.voxels_changed, rep.eye_dist_mm, rep.blur_sigma_mm);
    if (radveil_volume_save(vol, "/tmp/radveil_capi_out.nii.gz", 1)) {
        fprintf(stderr, "save: %s\n", radveil_last_error()); return 1;
    }
    radveil_volume_free(vol);
    printf("C API smoke test OK\n");
    return 0;
}
