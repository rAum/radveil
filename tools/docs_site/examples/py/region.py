import radveil

report = radveil.anonymization_mask("t1.nii.gz", mask_path="t1_region.nii.gz")
mask = report["mask"]  # uint8 (nx, ny, nz), 1 = voxel deface_auto would rewrite
print(mask.shape, int(mask.sum()), report["voxels_changed"])
