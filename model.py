import jax
import jax.numpy as jnp
from flax import nnx

class PatchEmbedding(nnx.Module):
  def __init__(self, in_channels, out_channels, kernel_size=4, stride=4, is_first=True, rngs=nnx.Rngs(0)):
    super().__init__()
    self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), rngs=rngs)
    if is_first:
      self.layer_norm = nnx.LayerNorm(out_channels, rngs=rngs)
      self.layer = nnx.Sequential(self.conv, self.layer_norm)
    else:
      self.layer_norm = nnx.LayerNorm(in_channels, rngs=rngs)
      self.layer = nnx.Sequential(self.layer_norm, self.conv)
  def __call__(self, X):
    return self.layer(X)

class ConvNextBlock(nnx.Module):
  def __init__(self, in_channels, out_channels, rngs=nnx.Rngs(0)):
    super().__init__()
    self.conv = nnx.Conv(in_channels, in_channels, kernel_size=(7,7), feature_group_count=in_channels, rngs=rngs)
    self.layer_norm = nnx.LayerNorm(in_channels, rngs=rngs)
    self.expand = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), rngs=rngs)
    self.act = nnx.gelu
    self.contract = nnx.Conv(out_channels, in_channels, kernel_size=(1, 1), rngs=rngs)
    self.gamma = nnx.Param(jnp.ones((in_channels,)))
  def __call__(self, X):
    output = self.conv(X)
    output = self.layer_norm(output)
    output = self.expand(output)
    output = self.act(output, approximate=False)
    output = self.contract(output)
    return X + output*self.gamma # residual

class DecoderBlock(nnx.Module):
  def __init__(self, in_channels, out_channels, rngs=nnx.Rngs(0)):
    super().__init__()
    self.conv = nnx.Conv(in_channels+out_channels, out_channels, kernel_size=(3,3), rngs=rngs)
    self.layer_norm = nnx.LayerNorm(out_channels, rngs=rngs)
  def __call__(self, X, encoder_input):
    X = jax.image.resize(X, (X.shape[0], *encoder_input.shape[1:3], X.shape[-1]), "bilinear")
    concat = jnp.concatenate([X, encoder_input], axis=-1) # channel axis
    return nnx.gelu(self.layer_norm(self.conv(concat)))


class ConvNextModel(nnx.Module):
  def __init__(self, depth=[3, 3, 9, 3], channels=[96, 192, 384, 768], rngs=nnx.Rngs(0)):
    super().__init__()
    self.stems = nnx.List([PatchEmbedding(3, channels[0], 4, 4, is_first=True, rngs=rngs) if i == 0 else PatchEmbedding(channels[i-1], channels[i], 2, 2, is_first=False, rngs=rngs) for i in range(4)])
    self.blocks = nnx.List([nnx.Sequential(
        *[ConvNextBlock(channels[i], 4*channels[i], rngs=rngs) for j in range(depth[i])]
    ) for i in range(4)])
    self.decoder_blocks = nnx.List([DecoderBlock(channels[3-i], channels[2-i], rngs=rngs) for i in range(3)])
    self.final = nnx.Conv(channels[0], 1, kernel_size=(1,1), rngs=rngs)
  def __call__(self, X):
    encoder_inputs = []
    # encoder
    for stem, block in zip(self.stems, self.blocks):
      X = stem(X)
      X = block(X)
      encoder_inputs.append(X)

    # decoder
    output = encoder_inputs[-1]
    for i in range(3):
      output = self.decoder_blocks[i](output, encoder_inputs[2-i])
    output = self.final(output)
    output = jax.image.resize(output, (output.shape[0], 4*output.shape[1], 4*output.shape[2], output.shape[3]), "bilinear")
    return jax.nn.softplus(output) + 1e-5