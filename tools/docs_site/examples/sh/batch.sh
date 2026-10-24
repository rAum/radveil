mkdir -p defaced
for f in cohort/*.nii.gz; do
  radveil auto "$f" "defaced/$(basename "$f" .nii.gz)_defaced.nii.gz" --mode adaptive \
    || echo "$f: refused or failed (exit $?)" >&2
done
