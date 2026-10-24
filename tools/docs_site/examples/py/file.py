import json
import radveil

report = radveil.deface_auto(
    "t1.nii.gz", "t1_defaced.nii.gz",
    mode="adaptive",
    region="face+ears",           # the pinna is biometric too
    ear_mode="cut",               # its own mode: blur leaves part of the pinna
    mask_path="t1_mask.nii.gz",   # exactly which voxels changed (uint8 NIfTI)
    qa_dir="qa",                  # before/after renders for review
)
for roi in report["rois"]:
    print(roi["label"], roi["voxels"], roi.get("detected"))
json.dump(report, open("report.json", "w"), indent=2)  # JSON-ready
