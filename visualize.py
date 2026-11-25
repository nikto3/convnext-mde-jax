import argparse
import os
from data import val_ds
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import load_checkpoint, forward

parser = argparse.ArgumentParser(description="Visualize depth predictions")
parser.add_argument("--pretrained-ckpt", type=str, required=True, help="Path to pretrained checkpoint")
parser.add_argument("--scratch-ckpt", type=str, required=True, help="Path to from-scratch checkpoint")
parser.add_argument("--num-samples", type=int, default=4, help="Number of validation samples")
parser.add_argument("--output", type=str, default="comparison", help="Output folder path")

args = parser.parse_args()

def unnormalize(image):
    mean=jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
    std=jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)
    return image*std + mean 
    
def visualize():
    # Load pretrained version
    if not os.path.exists(args.pretrained_ckpt):
        print(f"Path {args.pretrained_ckpt} cannot be found.")
        exit(-1)
    elif not os.path.exists(args.scratch_ckpt):
        print(f"Path {args.scratch_ckpt} cannot be found.")
        exit(-1)
    
    pretrained_model = load_checkpoint(os.path.abspath(args.pretrained_ckpt))
    scratch_model    = load_checkpoint(os.path.abspath(args.scratch_ckpt))
    
    os.makedirs(args.output, exist_ok=True)
    for i, batch in enumerate(val_ds.take(args.num_samples).iter(batch_size=1)):
        image = jnp.array(batch["image"])
        depth_map = jnp.array(batch["map"])
        pretrained_pred = forward(pretrained_model, image)
        scratch_pred = forward(scratch_model, image)
        image = jnp.clip(unnormalize(image), min=0.0, max=1.0)
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), layout="constrained")
        
        vmin = depth_map.min()
        vmax = depth_map.max()
        ax[0].imshow(image[0])
        ax[1].imshow(depth_map[0, ..., 0], cmap="magma", vmin=vmin, vmax=vmax)
        ax[2].imshow(pretrained_pred[0, ..., 0], cmap="magma", vmin=vmin, vmax=vmax)
        ax[3].imshow(scratch_pred[0, ..., 0], cmap="magma", vmin=vmin, vmax=vmax)
        
        for a in ax: a.axis("off")
        
        for j, title in enumerate(["Image", "Ground Truth", "Pretrained Model Prediction", "From Scratch Model Prediction"]):
            ax[j].set_title(title)
        fig.savefig(os.path.join(args.output, f"comparison_{i}.png"), dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    visualize()