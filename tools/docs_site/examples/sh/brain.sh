radveil deskull t1.nii.gz -o t1_brain.nii.gz --mask-out t1_brain_mask.nii.gz

# --brain-guard: never modify the brain or anything within 3 mm of it (~3 s).
# --brain-check: exit status 3 if a brain voxel was modified (output still written).
radveil auto t1.nii.gz t1_defaced.nii.gz --brain-guard --brain-check
