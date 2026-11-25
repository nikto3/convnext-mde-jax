import argparse
from utils import load_checkpoint, forward
import numpy as np
from data import val_ds
import pandas as pd
import jax.numpy as jnp
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a depth model on NYU Depth v2 validation set")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Orbax checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick CPU tests")
    parser.add_argument("--output", type=str, default="metrics", help="Optional file name for saving calculated metrics")
    return parser.parse_args()
  
def silog_fn(preds, labels):
    valid = ((labels > 1e-3) & np.isfinite(labels))
    p = preds[valid]
    g = labels[valid]
    d = np.log(p) - np.log(g)
    return  np.sqrt(np.maximum(np.mean(d**2) - 0.85 * (np.mean(d) ** 2), 1e-6))
  

def absrel_fn(preds, labels):
    valid = ((labels > 1e-3) & np.isfinite(labels))
    p = preds[valid]
    g = labels[valid]
    return np.mean(np.abs(p-g)/g) # per image

def rmse_fn(preds, labels):
    valid = ((labels > 1e-3) & np.isfinite(labels))
    p = preds[valid]
    g = labels[valid]
    diff = (p-g)
    return np.sqrt(np.mean(diff**2)) # per image 

def delta_fn(preds, labels):
    valid = ((labels > 1e-3) & np.isfinite(labels))
    p = preds[valid]
    g = labels[valid]
    ratio = np.maximum(p/g, g/p)
    delta = np.mean(ratio < 1.25)
    delta_2 = np.mean(ratio < 1.25**2)
    delta_3 = np.mean(ratio < 1.25**3)
    return (delta, delta_2, delta_3)
    
def eval(args):
    if not os.path.exists(args.ckpt):
        print(f"Path {args.ckpt} cannot be found.")
        exit(-1)
        
    model = load_checkpoint(os.path.abspath(args.ckpt))
    silog, absrel, rmse, delta, delta_2, delta_3 = [], [], [], [], [], []
    ds = val_ds.take(args.max_samples) if args.max_samples else val_ds
    for batch in ds.iter(batch_size=args.batch_size):
        images = jnp.array(batch["image"])
        depth_map = jnp.array(batch["map"])
        preds = forward(model, images)
        for p, g in zip(preds, depth_map):
            p, g = jnp.squeeze(p), jnp.squeeze(g)
            silog.append(silog_fn(p, g))
            absrel.append(absrel_fn(p, g))
            rmse.append(rmse_fn(p, g))
            deltas = delta_fn(p, g)
            delta.append(deltas[0])
            delta_2.append(deltas[1])
            delta_3.append(deltas[2])
        
    metrics = {
        "SiLog": [np.mean(silog)],
        "AbsRel": [np.mean(absrel)],
        "RMSE": [np.mean(rmse)],
        "Delta": [np.mean(delta)],
        "Delta^2": [np.mean(delta_2)],
        "Delta^3": [np.mean(delta_3)]
    }
    df = pd.DataFrame.from_dict(metrics)
    df.to_excel(f"{args.output}.xlsx", index=False)

if __name__ == "__main__":
    args = parse_args()
    eval(args)
    