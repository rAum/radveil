import radveil

rep = radveil.deskull("t1.nii.gz", "t1_brain.nii.gz", mask_path="t1_brain_mask.nii.gz")
print(f"brain {rep['brain_volume_ml']:.0f} ml")

# brain_guard: never modify the brain or anything within 3 mm of it (opt-in,
# ~3 s per volume); brain_check: report what happened, from the same extraction.
report = radveil.deface_auto("t1.nii.gz", "t1_defaced.nii.gz", brain_guard=True, brain_check=True)
assert report["brain"]["voxels_modified"] == 0, report["brain"]
print("brain untouched, margin", report["brain"]["margin_mm"], "mm")
