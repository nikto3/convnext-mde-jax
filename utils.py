from flax import nnx 
from model import ConvNextModel
from orbax.checkpoint import v1 as ocp
import jax.numpy as jnp
def load_checkpoint(checkpoint):
    m = ConvNextModel()
    state = nnx.state(ConvNextModel())
    restored_state = ocp.load(checkpoint, state)
    nnx.update(m, restored_state)
    return m
    
@nnx.jit
def forward(model, inputs):
    return model(inputs)
    
def load_pretrained(seed):
    import timm
    def assign(param, arr):
        assert param[...].shape == arr.shape, f"Param {param} has {param[...].shape}, and arr has {arr.shape} shape"
        param[...] = jnp.array(arr)
    pt_model = timm.create_model("convnext_tiny", pretrained=True)
    pt = {k: v.detach().cpu().numpy() for k, v in pt_model.state_dict().items()}
    model = ConvNextModel(rngs=nnx.Rngs(seed))
    assign(model.stems[0].conv.kernel, pt["stem.0.weight"].transpose(2,3,1,0))
    assign(model.stems[0].conv.bias, pt["stem.0.bias"])
    assign(model.stems[0].layer_norm.scale, pt["stem.1.weight"])
    assign(model.stems[0].layer_norm.bias, pt["stem.1.bias"])
    for s in range(1, 4):
        assign(model.stems[s].conv.kernel, pt[f"stages.{s}.downsample.1.weight"].transpose(2, 3, 1, 0))
        assign(model.stems[s].conv.bias, pt[f"stages.{s}.downsample.1.bias"])
        assign(model.stems[s].layer_norm.scale, pt[f"stages.{s}.downsample.0.weight"])
        assign(model.stems[s].layer_norm.bias, pt[f"stages.{s}.downsample.0.bias"])

    for s, stage in enumerate(model.blocks):
        for b, blk in enumerate(stage.layers):
            prefix = f"stages.{s}.blocks.{b}"
            assign(blk.conv.kernel, pt[f"{prefix}.conv_dw.weight"].transpose(2,3,1,0))
            assign(blk.conv.bias, pt[f"{prefix}.conv_dw.bias"])

            assign(blk.layer_norm.scale, pt[f"{prefix}.norm.weight"])
            assign(blk.layer_norm.bias, pt[f"{prefix}.norm.bias"])

            assign(blk.expand.kernel, pt[f"{prefix}.mlp.fc1.weight"][None, None, ...].transpose(0, 1, 3, 2))
            assign(blk.expand.bias, pt[f"{prefix}.mlp.fc1.bias"])

            assign(blk.contract.kernel, pt[f"{prefix}.mlp.fc2.weight"][None, None, ...].transpose(0, 1, 3, 2))
            assign(blk.contract.bias, pt[f"{prefix}.mlp.fc2.bias"])

            assign(blk.gamma, pt[f"{prefix}.gamma"])
    return model