import radveil

report = radveil.deface_auto("t1.nii.gz", "t1_defaced.nii.gz", mode="adaptive")
print(report["voxels_changed"], "voxels changed;", report["residual_detections"], "faces left")
