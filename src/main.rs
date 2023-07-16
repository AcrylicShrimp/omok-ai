mod plot;
mod trainer;
mod utils;

use burn_autodiff::ADBackendDecorator;
use burn_tch::{TchBackend, TchDevice};
use trainer::Trainer;

fn main() {
    #[cfg(not(target_os = "macos"))]
    let device = TchDevice::Cuda(0);
    #[cfg(target_os = "macos")]
    let device = TchDevice::Cpu;

    let mut train = Trainer::<ADBackendDecorator<TchBackend<f32>>>::new(device.clone());
    train.train(device, 10_000);
}
