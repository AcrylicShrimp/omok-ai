use crate::{SeparableConv2d, SeparableConv2dConfig};
use burn::{
    config::Config,
    module::Module,
    nn::{conv::Conv2dPaddingConfig, BatchNorm, BatchNormConfig, Initializer, GELU},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct ResidualBlockConfig {
    /// The number of channels.
    pub channels: usize,
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters.
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    pub conv0: SeparableConv2d<B>,
    pub bn0: BatchNorm<B, 2>,
    pub activation: GELU,
    pub conv1: SeparableConv2d<B>,
    pub bn1: BatchNorm<B, 2>,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self) -> ResidualBlock<B> {
        let conv0 = self.conv0_config().init();
        let bn0 = self.bn0_config().init();
        let activation = GELU::new();
        let bn1 = self.bn1_config().init();
        let conv1 = self.conv1_config().init();

        ResidualBlock {
            conv0,
            bn0,
            activation,
            conv1,
            bn1,
        }
    }

    pub fn init_with<B: Backend>(&self, record: ResidualBlockRecord<B>) -> ResidualBlock<B> {
        ResidualBlock {
            conv0: self.conv0_config().init_with(record.conv0),
            bn0: self.bn0_config().init_with(record.bn0),
            activation: GELU::new(),
            conv1: self.conv1_config().init_with(record.conv1),
            bn1: self.bn1_config().init_with(record.bn1),
        }
    }

    pub fn conv0_config(&self) -> SeparableConv2dConfig {
        SeparableConv2dConfig {
            channels: [self.channels, self.channels],
            kernel_size: self.kernel_size,
            stride: self.stride,
            dilation: self.dilation,
            padding: Conv2dPaddingConfig::Same,
            bias: self.bias,
            initializer: self.initializer.clone(),
        }
    }

    pub fn bn0_config(&self) -> BatchNormConfig {
        BatchNormConfig {
            num_features: self.channels,
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }

    pub fn conv1_config(&self) -> SeparableConv2dConfig {
        SeparableConv2dConfig {
            channels: [self.channels, self.channels],
            kernel_size: self.kernel_size,
            stride: self.stride,
            dilation: self.dilation,
            padding: Conv2dPaddingConfig::Same,
            bias: self.bias,
            initializer: self.initializer.clone(),
        }
    }

    pub fn bn1_config(&self) -> BatchNormConfig {
        BatchNormConfig {
            num_features: self.channels,
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv0.forward(input.clone());
        let x = self.bn0.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        x + input
    }
}
