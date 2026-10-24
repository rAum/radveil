use radveil_core::{brain_mask, deface_auto, AutoParams, DeskullParams, MediaPipeDetector, Volume};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Brain extraction on its own (SynthStrip, pure Rust).
    let vol = Volume::load("t1.nii.gz")?;
    let (brain, rep) = brain_mask(&vol, &DeskullParams::default())?;
    println!(
        "brain {:.0} ml, {} voxels",
        rep.brain_volume_ml,
        brain.iter().filter(|&&b| b == 1).count()
    );

    // Or around anonymization: the guard never modifies the brain (opt-in,
    // ~3 s per volume); the check reports on it from the same extraction.
    let mut vol = Volume::load("t1.nii.gz")?;
    let params = AutoParams {
        brain_guard: true,
        brain_check: true,
        ..Default::default()
    };
    let report = deface_auto(&mut vol, &mut MediaPipeDetector::default(), &params)?;
    let check = report.brain.expect("brain_check was requested");
    assert_eq!(check.voxels_modified, 0, "anonymization touched the brain");
    Ok(())
}
