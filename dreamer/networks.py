# SOURCE: NaturalDreamer (https://github.com/InexperiencedMe/NaturalDreamer) — da adattare
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        return self.recurrent(self.activation(self.linear(torch.cat((latentState, action), -1))), recurrentState)


class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)

    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)

    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))


class EncoderConv(nn.Module):
    def __init__(self, inputShape, outputSize, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()
        channels, height, width = inputShape
        self.outputSize = outputSize

        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(channels,            self.config.depth*1, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Flatten(),
            nn.Linear(self.config.depth*8*(height // (self.config.stride ** 4))*(width // (self.config.stride ** 4)), outputSize), activation)

    def forward(self, x):
        return self.convolutionalNet(x).view(-1, self.outputSize)


class DecoderConv(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = outputShape
        activation = getattr(nn, self.config.activation)()

        self.network = nn.Sequential(
            nn.Linear(inputSize, self.config.depth*32),
            nn.Unflatten(1, (self.config.depth*32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth*32, self.config.depth*4, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*4,  self.config.depth*2, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*2,  self.config.depth*1, self.config.kernelSize + 1, self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*1,  self.channels,       self.config.kernelSize + 1, self.config.stride))

    def forward(self, x):
        return self.network(x)


class MLPEncoder(nn.Module):
    """Feedforward encoder replacing EncoderConv for vector observations.

    Uses sequentialModel1D — same pattern as PriorNet/PosteriorNet.
    No convolutional layers; processes flat observation vectors directly.

    Args:
        obs_dim: Input observation dimension (e.g., 23 for LimoCustomEnv).
        output_size: Encoded representation dimension (e.g., 128).
        config: Config object with hiddenSize, numLayers, activation.

    Example:
        >>> enc = MLPEncoder(23, 128, config)
        >>> out = enc(torch.randn(4, 23))
        >>> out.shape
        torch.Size([4, 128])
    """
    def __init__(self, obs_dim, output_size, config):
        super().__init__()
        self.output_size = output_size
        self.net = sequentialModel1D(
            obs_dim,
            [config.hiddenSize] * config.numLayers,
            output_size,
            config.activation)

    def forward(self, x):
        """Encode observation vector.

        Args:
            x: Tensor of shape (..., obs_dim). Leading dims are preserved.

        Returns:
            Tensor of shape (..., output_size).

        Example:
            >>> enc = MLPEncoder(23, 128, config)
            >>> enc(torch.randn(2, 4, 23)).shape
            torch.Size([2, 4, 128])
        """
        return self.net(x.view(-1, x.shape[-1])).view(*x.shape[:-1], self.output_size)


class MLPDecoder(nn.Module):
    """Feedforward decoder replacing DecoderConv for vector observations.

    Outputs raw mean values for Normal(mean, 1) reconstruction loss.
    No activation on final layer — same pattern as RewardModel/Critic.

    Args:
        input_size: Input dimension (typically fullStateSize = recurrent + latent).
        obs_dim: Output observation dimension to reconstruct (e.g., 23).
        config: Config object with hiddenSize, numLayers, activation.

    Example:
        >>> dec = MLPDecoder(512, 23, config)
        >>> out = dec(torch.randn(4, 512))
        >>> out.shape
        torch.Size([4, 23])
    """
    def __init__(self, input_size, obs_dim, config):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = sequentialModel1D(
            input_size,
            [config.hiddenSize] * config.numLayers,
            obs_dim,
            config.activation)

    def forward(self, x):
        """Decode latent state back to observation reconstruction.

        Args:
            x: Tensor of shape (..., input_size). Leading dims preserved.

        Returns:
            Tensor of shape (..., obs_dim) — raw mean values.

        Example:
            >>> dec = MLPDecoder(512, 23, config)
            >>> dec(torch.randn(2, 4, 512)).shape
            torch.Size([2, 4, 23])
        """
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()
        actionSize *= 2
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias", ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))

    def forward(self, x, training=False):
        logStdMin, logStdMax = -5, 2
        mean, logStd = self.network(x).chunk(2, dim=-1)
        logStd = logStdMin + (logStdMax - logStdMin)/2*(torch.tanh(logStd) + 1)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.actionScale + self.actionBias
        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale*(1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


if __name__ == '__main__':
    # ── Self-check: MLPEncoder and MLPDecoder ──────────────────────────
    print('=== SELF-CHECK: networks.py (MLPEncoder / MLPDecoder) ===')

    class _MockConfig:
        pass

    cfg = _MockConfig()
    cfg.hiddenSize = 256
    cfg.numLayers = 2
    cfg.activation = 'ELU'

    obs_dim = 23
    output_size = 128
    full_state_size = 512  # 256 recurrent + 256 latent (latentLength*latentClasses = 16*16)

    # Test 1: MLPEncoder
    encoder = MLPEncoder(obs_dim, output_size, cfg)
    dummy_obs = torch.randn(4, obs_dim)
    encoded = encoder(dummy_obs)
    expected_shape = (4, output_size)
    passed_1 = encoded.shape == expected_shape
    print(f'MLPEncoder: input (4, {obs_dim}) → output {tuple(encoded.shape)} '
          f'(expected {expected_shape}) — {"PASS" if passed_1 else "FAIL"}')
    assert passed_1, f'Encoder shape mismatch: got {encoded.shape}, expected {expected_shape}'

    # Test 2: MLPDecoder
    decoder = MLPDecoder(full_state_size, obs_dim, cfg)
    dummy_state = torch.randn(4, full_state_size)
    decoded = decoder(dummy_state)
    expected_shape = (4, obs_dim)
    passed_2 = decoded.shape == expected_shape
    print(f'MLPDecoder: input (4, {full_state_size}) → output {tuple(decoded.shape)} '
          f'(expected {expected_shape}) — {"PASS" if passed_2 else "FAIL"}')
    assert passed_2, f'Decoder shape mismatch: got {decoded.shape}, expected {expected_shape}'

    # Test 3: MLPEncoder preserves batch dims with 3D input
    dummy_obs_3d = torch.randn(2, 4, obs_dim)
    encoded_3d = encoder(dummy_obs_3d)
    expected_shape_3d = (2, 4, output_size)
    passed_3 = encoded_3d.shape == expected_shape_3d
    print(f'MLPEncoder 3D: input (2, 4, {obs_dim}) → output {tuple(encoded_3d.shape)} '
          f'(expected {expected_shape_3d}) — {"PASS" if passed_3 else "FAIL"}')
    assert passed_3, f'Encoder 3D shape mismatch: got {encoded_3d.shape}, expected {expected_shape_3d}'

    # Test 4: MLPDecoder preserves batch dims with 3D input
    dummy_state_3d = torch.randn(2, 4, full_state_size)
    decoded_3d = decoder(dummy_state_3d)
    expected_shape_3d = (2, 4, obs_dim)
    passed_4 = decoded_3d.shape == expected_shape_3d
    print(f'MLPDecoder 3D: input (2, 4, {full_state_size}) → output {tuple(decoded_3d.shape)} '
          f'(expected {expected_shape_3d}) — {"PASS" if passed_4 else "FAIL"}')
    assert passed_4, f'Decoder 3D shape mismatch: got {decoded_3d.shape}, expected {expected_shape_3d}'

    print(f'All tests: {"PASS" if all([passed_1, passed_2, passed_3, passed_4]) else "FAIL"}')
    print(f'(Note: roadmap §6.2 says input_size=272 (256+16) but dreamer.py __init__ '
          f'gives fullStateSize = recurrentSize + latentLength*latentClasses '
          f'= 256 + 16*16 = 512. Used 512.)')
