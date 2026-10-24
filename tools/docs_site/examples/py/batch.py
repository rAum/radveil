import radveil

# Threads in this process by default (the Rust core releases the GIL);
# executor="processes" isolates each volume in a worker process instead.
reports = radveil.deface_batch("cohort/*.nii.gz", output_dir="defaced", mode="adaptive", workers=2)
for r in reports:
    print(r["input"], r.get("error") or f"{r['voxels_changed']} voxels")
