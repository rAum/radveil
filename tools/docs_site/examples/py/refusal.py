import radveil

try:
    radveil.deface_auto("t1_defaced.nii.gz", "twice.nii.gz")
except RuntimeError as refusal:  # nothing was written
    print("refused:", str(refusal)[:60])
