use super::Network;
use crate::NetworkConfig;
use burn::{module::Module, optim::AdamConfig, tensor::backend::Backend};

pub struct AgentModel<B: Backend> {
    pub network: Network<B>,
    pub optimizer: AdamConfig,
}

impl<B: Backend> AgentModel<B> {
    pub fn new(device: B::Device) -> Self {
        let network = NetworkConfig {
            input_channels: 2,
            residual_blocks: 5,
            residual_channels: 16,
            residual_kernel_size: [3, 3],
            residual_stride: [1, 1],
            residual_dilation: [1, 1],
            value_channels: 2,
            value_fc0_features: 32,
            value_fc1_features: 32,
            policy_channels: 16,
            policy_fc0_features: 64,
            policy_fc1_features: 64,
        }
        .init()
        .fork(&device);

        Self {
            network,
            optimizer: AdamConfig::new(),
        }
    }
}
