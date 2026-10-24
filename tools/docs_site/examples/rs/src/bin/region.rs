use radveil_core::{anonymization_mask, AutoParams, MediaPipeDetector, Volume};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vol = Volume::load("t1.nii.gz")?;
    let report = anonymization_mask(
        &vol,
        &mut MediaPipeDetector::default(),
        &AutoParams::default(),
    )?;
    let mask = report
        .mask
        .expect("anonymization_mask always returns the mask");
    let n = mask.iter().filter(|&&m| m == 1).count();
    println!(
        "{n} voxels would be rewritten (report: {})",
        report.deface.voxels_changed
    );
    // Save it on the same grid, e.g. for review in a viewer.
    Volume::from_array(mask.mapv(f32::from), vol.affine).save("t1_region.nii.gz")?;
    Ok(())
}
