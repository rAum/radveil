import numpy as np
import radveil

def my_detector(rgb: np.ndarray, try_rotations: bool):
    """Return None, a (68, 2) array in the 68-point (iBUG) order, or (array, score)."""
    return radveil.detect_landmarks(rgb, try_rotations=try_rotations)

report = radveil.deface_auto("t1.nii.gz", "t1_defaced.nii.gz", detector=my_detector)
print(report["detector"], report["views_detected"])
