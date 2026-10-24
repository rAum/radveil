import nibabel as nib
import numpy as np
import radveil

img = nib.load("t1.nii.gz")  # or a DICOM series read with pydicom / SimpleITK
vol = radveil.Volume.from_array(np.asarray(img.dataobj, dtype=np.float32), img.affine)
report = radveil.deface_volume(vol, mode="adaptive")  # modifies vol in place
nib.save(nib.Nifti1Image(vol.data(), img.affine), "t1_defaced.nii.gz")
print(report["views_detected"], "views,", report["verify_rounds"], "verification rounds")
