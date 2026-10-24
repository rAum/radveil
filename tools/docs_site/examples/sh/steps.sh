radveil info t1.nii.gz                               # shape, spacing, orientation, affine
radveil render t1.nii.gz -o renders --yaw=-20,0,20   # PNG views + camera JSON
# 68 world-space (RAS+ mm) points from any source: {"points": [[x, y, z] | null, ...]}
radveil deface t1.nii.gz t1_defaced.nii.gz --landmarks3d points.json --mode adaptive
