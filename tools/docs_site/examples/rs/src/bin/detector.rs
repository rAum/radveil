use radveil_core::{deface_auto, AutoParams, Detection, FaceDetector, Volume};

/// Any 2D landmark model can drive the pipeline: return 68 points in
/// the standard 68-point (iBUG) order (pixel coordinates of the rendered view), or None.
struct MyDetector;

impl FaceDetector for MyDetector {
    fn detect(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        _try_rotations: bool,
    ) -> Result<Option<Detection>, String> {
        let _ = (rgb, w, h); // run your model here
        Ok(None)
    }
}

fn main() {
    let mut vol = Volume::load("t1.nii.gz").expect("load");
    match deface_auto(&mut vol, &mut MyDetector, &AutoParams::default()) {
        Ok(r) => println!("{} voxels changed", r.deface.voxels_changed),
        Err(refusal) => eprintln!("refused, volume untouched: {refusal}"),
    }
}
