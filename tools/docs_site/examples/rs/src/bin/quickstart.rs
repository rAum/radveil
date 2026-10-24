use radveil_core::{deface_auto, AutoParams, DefaceMode, DefaceParams, MediaPipeDetector, Volume};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vol = Volume::load("t1.nii.gz")?;
    let params = AutoParams {
        deface: DefaceParams {
            mode: DefaceMode::Adaptive,
            ..Default::default()
        },
        ..Default::default()
    };
    let report = deface_auto(&mut vol, &mut MediaPipeDetector::default(), &params)?;
    println!(
        "{} voxels changed, faces left: {:?}",
        report.deface.voxels_changed, report.residual_detections
    );
    vol.save("t1_defaced.nii.gz")?;
    Ok(())
}
