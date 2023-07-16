use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, Conv2dPaddingConfig},
        Initializer,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct SeparableConv2dConfig {
    /// The number of channels.
    pub channels: [usize; 2],
    /// The size of the kernel.
    pub kernel_size: [usize; 2],
    /// The stride of the convolution.
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    /// Spacing between kernel elements.
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    /// The padding configuration.
    #[config(default = "Conv2dPaddingConfig::Valid")]
    pub padding: Conv2dPaddingConfig,
    /// If bias should be added to the output.
    #[config(default = true)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters.
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct SeparableConv2d<B: Backend> {
    pub depthwise: Conv2d<B>,
    pub pointwise: Conv2d<B>,
}

impl SeparableConv2dConfig {
    pub fn init<B: Backend>(&self) -> SeparableConv2d<B> {
        let depthwise = self.depthwise_config().init();
        let pointwise = self.pointwise_config().init();

        SeparableConv2d {
            depthwise,
            pointwise,
        }
    }

    pub fn init_with<B: Backend>(&self, record: SeparableConv2dRecord<B>) -> SeparableConv2d<B> {
        SeparableConv2d {
            depthwise: self.depthwise_config().init_with(record.depthwise),
            pointwise: self.pointwise_config().init_with(record.pointwise),
        }
    }

    pub fn depthwise_config(&self) -> Conv2dConfig {
        Conv2dConfig {
            channels: [1, self.channels[0]],
            kernel_size: self.kernel_size,
            stride: self.stride,
            dilation: self.dilation,
            groups: self.channels[0],
            padding: self.padding.clone(),
            bias: self.bias,
            initializer: self.initializer.clone(),
        }
    }

    pub fn pointwise_config(&self) -> Conv2dConfig {
        Conv2dConfig {
            channels: [self.channels[0], self.channels[1]],
            kernel_size: [1, 1],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
            padding: Conv2dPaddingConfig::Same,
            bias: self.bias,
            initializer: self.initializer.clone(),
        }
    }
}

impl<B: Backend> SeparableConv2d<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise.forward(input);
        let x = self.pointwise.forward(x);
        x
    }
}
