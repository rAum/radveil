# The CLI always anonymizes. The region-only escape hatch is API-only;
# --mask-out writes the region alongside the anonymized output.
radveil auto t1.nii.gz t1_defaced.nii.gz --mode adaptive --mask-out t1_region.nii.gz
