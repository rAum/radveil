# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate crates/radveil-core/models/*.rvnn from the MediaPipe Face
Landmarker bundle. One-off developer tool; RadVeil itself never runs it.

    pip install tflite numpy tensorflow     # tensorflow only for --verify
    python tools/convert_mediapipe_models.py

Downloads face_landmarker.task (a zip of float16 TFLite models), exports the
face detector and the face-mesh model with tools/export_rvnn.py, and checks
each export against TensorFlow Lite's interpreter. After regenerating, update
crates/radveil-core/tests/data/nn_golden.bin if the models changed.
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "crates", "radveil-core", "models")
MODELS = {
    "face_detector.tflite": "blazeface_short_range.rvnn",
    "face_landmarks_detector.tflite": "face_mesh_v2.rvnn",
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        task = os.path.join(tmp, "face_landmarker.task")
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, task)
        print(f"  sha256 {sha256(task)}")
        with zipfile.ZipFile(task) as z:
            z.extractall(tmp)
        for src, dst in MODELS.items():
            out = os.path.join(OUT, dst)
            subprocess.check_call(
                [sys.executable, os.path.join(HERE, "export_rvnn.py"), "--verify",
                 os.path.join(tmp, src), out]
            )
            print(f"{dst}: sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
