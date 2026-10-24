if ! radveil auto t1_defaced.nii.gz twice.nii.gz; then
  echo "refused: nothing was written"   # exit status 1
fi
