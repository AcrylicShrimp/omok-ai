use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, Conv2dPaddingConfig},
        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, GELU,
    },
    record::Recorder,
    tensor::{
        activation::{log_softmax, softmax},
        backend::{ADBackend, Backend},
        Tensor,
    },
    train::{TrainOutput, TrainStep},
};
use environment::Environment;
use network_utils::{ResidualBlock, ResidualBlockConfig};

#[derive(Config)]
pub struct NetworkConfig {
    /// The number of input channels.
    pub input_channels: usize,

    /// The number of residual blocks.
    pub residual_blocks: usize,
    /// The number of channels for each residual block.
    pub residual_channels: usize,
    /// The size of the kernel for each residual block.
    pub residual_kernel_size: [usize; 2],
    /// The stride of the convolution for each residual block.
    #[config(default = "[1, 1]")]
    pub residual_stride: [usize; 2],
    /// Spacing between kernel elements for each residual block.
    #[config(default = "[1, 1]")]
    pub residual_dilation: [usize; 2],

    /// The number of channels for the value head convolution.
    pub value_channels: usize,
    /// The number of channels for the first fully connected layer of the value head.
    pub value_fc0_features: usize,
    /// The number of channels for the second fully connected layer of the value head.
    pub value_fc1_features: usize,

    /// The number of channels for the policy head convolution.
    pub policy_channels: usize,
    /// The number of channels for the first fully connected layer of the policy head.
    pub policy_fc0_features: usize,
    /// The number of channels for the second fully connected layer of the policy head.
    pub policy_fc1_features: usize,
}

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    pub head_conv: Conv2d<B>,
    pub head_bn: BatchNorm<B, 2>,
    pub head_activation: GELU,
    pub residual_blocks: Vec<ResidualBlockWithActivation<B>>,
    pub value_head: ValueHead<B>,
    pub policy_head: PolicyHead<B>,
}

impl NetworkConfig {
    pub fn init<B: Backend>(&self) -> Network<B> {
        let head_conv = Conv2dConfig {
            channels: [self.input_channels, self.residual_channels],
            kernel_size: [1, 1],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
            padding: Conv2dPaddingConfig::Same,
            bias: true,
            initializer: Initializer::Normal(0.0, 2.0 / f64::sqrt(self.input_channels as f64)),
        }
        .init();
        let head_bn = BatchNormConfig {
            num_features: self.residual_channels,
            epsilon: 1e-5,
            momentum: 0.1,
        }
        .init();
        let head_activation = GELU::new();
        let residual_blocks = (0..self.residual_blocks)
            .into_iter()
            .map(|_| ResidualBlockWithActivation {
                residual_block: ResidualBlockConfig {
                    channels: self.residual_channels,
                    kernel_size: self.residual_kernel_size,
                    stride: self.residual_stride,
                    dilation: self.residual_dilation,
                    bias: true,
                    initializer: Initializer::Normal(
                        0.0,
                        2.0 / f64::sqrt(
                            (self.residual_kernel_size[1]
                                * self.residual_kernel_size[0]
                                * self.residual_channels) as f64,
                        ),
                    ),
                }
                .init(),
                activation: GELU::new(),
            })
            .collect();
        let value_head = ValueHead {
            conv: Conv2dConfig {
                channels: [self.residual_channels, self.value_channels],
                kernel_size: [1, 1],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
                padding: Conv2dPaddingConfig::Same,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(self.residual_channels as f64),
                ),
            }
            .init(),
            conv_bn: BatchNormConfig {
                num_features: self.value_channels,
                epsilon: 1e-5,
                momentum: 0.1,
            }
            .init(),
            conv_activation: GELU::new(),
            fc0: LinearConfig {
                d_input: self.value_channels * Environment::BOARD_SIZE * Environment::BOARD_SIZE,
                d_output: self.value_fc0_features,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(
                        (self.value_channels * Environment::BOARD_SIZE * Environment::BOARD_SIZE)
                            as f64,
                    ),
                ),
            }
            .init(),
            fc0_bn: BatchNormConfig {
                num_features: self.value_fc0_features,
                epsilon: 1e-5,
                momentum: 0.1,
            }
            .init(),
            fc0_activation: GELU::new(),
            fc1: LinearConfig {
                d_input: self.value_fc0_features,
                d_output: self.value_fc1_features,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(self.value_fc0_features as f64),
                ),
            }
            .init(),
            fc1_bn: BatchNormConfig {
                num_features: self.value_fc1_features,
                epsilon: 1e-5,
                momentum: 0.1,
            }
            .init(),
            fc1_activation: GELU::new(),
            fc2: LinearConfig {
                d_input: self.value_fc1_features,
                d_output: 1,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(self.value_fc1_features as f64),
                ),
            }
            .init(),
        };
        let policy_head = PolicyHead {
            conv: Conv2dConfig {
                channels: [self.residual_channels, self.policy_channels],
                kernel_size: [1, 1],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
                padding: Conv2dPaddingConfig::Same,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(self.residual_channels as f64),
                ),
            }
            .init(),
            conv_bn: BatchNormConfig {
                num_features: self.policy_channels,
                epsilon: 1e-5,
                momentum: 0.1,
            }
            .init(),
            conv_activation: GELU::new(),
            fc0: LinearConfig {
                d_input: self.policy_channels * Environment::BOARD_SIZE * Environment::BOARD_SIZE,
                d_output: self.policy_fc0_features,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(
                        (self.policy_channels * Environment::BOARD_SIZE * Environment::BOARD_SIZE)
                            as f64,
                    ),
                ),
            }
            .init(),
            fc0_bn: BatchNormConfig {
                num_features: self.policy_fc0_features,
                epsilon: 1e-5,
                momentum: 0.1,
            }
            .init(),
            fc0_activation: GELU::new(),
            fc1: LinearConfig {
                d_input: self.policy_fc0_features,
                d_output: self.policy_fc1_features,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(self.policy_fc0_features as f64),
                ),
            }
            .init(),
            fc1_bn: BatchNormConfig {
                num_features: self.policy_fc1_features,
                epsilon: 1e-5,
                momentum: 0.1,
            }
            .init(),
            fc1_activation: GELU::new(),
            fc2: LinearConfig {
                d_input: self.policy_fc1_features,
                d_output: Environment::BOARD_SIZE * Environment::BOARD_SIZE,
                bias: true,
                initializer: Initializer::Normal(
                    0.0,
                    2.0 / f64::sqrt(self.policy_fc1_features as f64),
                ),
            }
            .init(),
        };

        Network {
            head_conv,
            head_bn,
            head_activation,
            residual_blocks,
            value_head,
            policy_head,
        }
    }
}

impl<B: Backend> Network<B> {
    pub fn train(
        &self,
        input: Tensor<B, 4>,
        value_target: Tensor<B, 1>,
        policy_target: Tensor<B, 2>,
    ) -> (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1>) {
        let x = self.forward(input);
        let (value, value_loss) = self.value_head.train(x.clone(), value_target);
        let (policy, policy_loss) = self.policy_head.train(x, policy_target);
        let loss = value_loss + policy_loss;
        (value, policy, loss)
    }

    pub fn infer(&self, input: Tensor<B, 4>) -> (Tensor<B, 1>, Tensor<B, 2>) {
        let x = self.forward(input);
        let value = self.value_head.infer(x.clone());
        let policy = self.policy_head.infer(x);
        (value, policy)
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.head_conv.forward(input);
        let x = self.head_bn.forward(x);
        let x = self.head_activation.forward(x);

        let mut x = x;
        for residual_block in &self.residual_blocks {
            x = residual_block.forward(x);
        }

        x
    }
}

impl<B: ADBackend>
    TrainStep<
        (Tensor<B, 4>, Tensor<B, 1>, Tensor<B, 2>),
        (Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1>),
    > for Network<B>
{
    fn step(
        &self,
        item: (Tensor<B, 4>, Tensor<B, 1>, Tensor<B, 2>),
    ) -> TrainOutput<(Tensor<B, 1>, Tensor<B, 2>, Tensor<B, 1>)> {
        let (value, policy, loss) = self.train(item.0, item.1, item.2);
        TrainOutput::new(self, loss.backward(), (value, policy, loss))
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlockWithActivation<B: Backend> {
    pub residual_block: ResidualBlock<B>,
    pub activation: GELU,
}

impl<B: Backend> ResidualBlockWithActivation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.residual_block.forward(input);
        let x = self.activation.forward(x);
        x
    }
}

/// Value head sub-network.
/// This sub-network is used to predict the value of the current state.
#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    pub conv: Conv2d<B>,
    pub conv_bn: BatchNorm<B, 2>,
    pub conv_activation: GELU,
    pub fc0: Linear<B>,
    pub fc0_bn: BatchNorm<B, 1>,
    pub fc0_activation: GELU,
    pub fc1: Linear<B>,
    pub fc1_bn: BatchNorm<B, 1>,
    pub fc1_activation: GELU,
    pub fc2: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
    pub fn train(&self, input: Tensor<B, 4>, target: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let x = self.forward(input);
        let loss = {
            let diff = x.clone() - target;
            let diff = diff.powf(2.0);
            diff.mean()
        };
        (x, loss)
    }

    pub fn infer(&self, input: Tensor<B, 4>) -> Tensor<B, 1> {
        self.forward(input)
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 1> {
        let x = self.conv.forward(input);
        let x = self.conv_bn.forward(x);
        let x = self.conv_activation.forward(x);
        let x = x.flatten::<2>(1, 3);
        let x = self.fc0.forward(x);
        let x = self
            .fc0_bn
            .forward({
                let shape = [x.dims()[0], x.dims()[1], 1];
                x.reshape(shape)
            })
            .flatten::<2>(1, 2);
        let x = self.fc0_activation.forward(x);
        let x = self.fc1.forward(x);
        let x = self
            .fc1_bn
            .forward({
                let shape = [x.dims()[0], x.dims()[1], 1];
                x.reshape(shape)
            })
            .flatten::<2>(1, 2);
        let x = self.fc1_activation.forward(x);
        let x = self.fc2.forward(x);
        let x = x.tanh();
        let x = x.flatten(0, 1);
        x
    }
}

/// Policy head sub-network.
/// This sub-network is used to predict the policy of the current state.
#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    pub conv: Conv2d<B>,
    pub conv_bn: BatchNorm<B, 2>,
    pub conv_activation: GELU,
    pub fc0: Linear<B>,
    pub fc0_bn: BatchNorm<B, 1>,
    pub fc0_activation: GELU,
    pub fc1: Linear<B>,
    pub fc1_bn: BatchNorm<B, 1>,
    pub fc1_activation: GELU,
    pub fc2: Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    pub fn train(&self, input: Tensor<B, 4>, target: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let x = self.forward(input);
        let loss = {
            let log_softmax = log_softmax(x.clone(), 1);
            let loss = (log_softmax * target).sum_dim(1);
            -loss.mean()
        };
        let x = softmax(x, 1);
        (x, loss)
    }

    pub fn infer(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.forward(input);
        let output = softmax(x, 1);
        output
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv.forward(input);
        let x = self.conv_bn.forward(x);
        let x = self.conv_activation.forward(x);
        let x = x.flatten::<2>(1, 3);
        let x = self.fc0.forward(x);
        let x = self
            .fc0_bn
            .forward({
                let shape = [x.dims()[0], x.dims()[1], 1];
                x.reshape(shape)
            })
            .flatten::<2>(1, 2);
        let x = self.fc0_activation.forward(x);
        let x = self.fc1.forward(x);
        let x = self
            .fc1_bn
            .forward({
                let shape = [x.dims()[0], x.dims()[1], 1];
                x.reshape(shape)
            })
            .flatten::<2>(1, 2);
        let x = self.fc1_activation.forward(x);
        let x = self.fc2.forward(x);
        x
    }
}
