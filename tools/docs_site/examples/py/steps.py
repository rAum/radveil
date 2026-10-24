import radveil

vol = radveil.Volume("t1.nii.gz")
views = [vol.render(yaw_deg=yaw) for yaw in (-20, 0, 20)]
landmarks = [radveil.detect_landmarks(v["rgb"]) for v in views]  # (68, 2) or None
points = radveil.lift_landmarks(views, landmarks)                # 68 x (x, y, z) | None
report = vol.deface(points, mode="adaptive")
vol.save("t1_defaced.nii.gz")
print(report["voxels_changed"], "voxels changed")
