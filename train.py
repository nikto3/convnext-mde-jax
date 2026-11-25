import argparse
from model import ConvNextModel
from flax import nnx
import optax
from orbax.checkpoint import v1 as ocp
from utils import load_pretrained
from data import transform_fn, train_pipeline, train_ds, val_ds
import time
import os 
import jax.numpy as jnp

def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny U-Net on NYU Depth v2")
    parser.add_argument("--pretrained", action="store_true", help="Load ImageNet-pretrained timm ConvNeXt-Tiny encoder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training and validation batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Linear warmup epochs")
    parser.add_argument("--ckpt-dir", type=str, default="best", help="Directory to save best Orbax checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def loss_fn(model, inputs, labels):
    preds = model(inputs)
    valid = ((labels > 1e-3) & jnp.isfinite(labels))
    mask = valid.astype(jnp.float32)
    safe_labels = jnp.where(valid, labels, 1.0)
    d = (jnp.log(preds) - jnp.log(safe_labels)) * mask
    _lambda = 0.85
    n = jnp.sum(mask) + 1e-6
    loss = jnp.sqrt(jnp.maximum(jnp.sum(d**2)/n - (jnp.sum(d)**2 * _lambda)/n**2, 1e-6))
    return loss, preds

@nnx.jit
def train_step(model, optimizer, inputs, labels):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, inputs, labels)
    optimizer.update(model, grads)
    return loss

@nnx.jit
def eval_step(model, inputs, labels):
    loss, _ = loss_fn(model, inputs, labels)
    return loss

def train_model(args):
    if args.pretrained:
        model = load_pretrained(args.seed)
    else:
        model = ConvNextModel(rngs=nnx.Rngs(args.seed))
    epochs = args.epochs
    lr = args.lr 
    batch_size = args.batch_size
    steps_per_epoch = 3100 // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    weight_decay = args.weight_decay
    schedule = optax.schedules.warmup_cosine_decay_schedule(1e-6, peak_value=lr, warmup_steps=warmup_steps, decay_steps=total_steps, end_value=1e-6)
    tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    avg_train, avg_val = [], []
    best_val_loss = float("+inf")
    ds_train = train_ds.shuffle(args.seed, buffer_size=256).map(lambda x: transform_fn(x, train_pipeline))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    for epoch in range(1, epochs+1):
        ds_train.set_epoch(epoch)
        train = []
        start = time.time()
        for batch in ds_train.iter(batch_size=batch_size, drop_last_batch=True):
            images = jnp.array(batch["image"])
            maps = jnp.array(batch["map"])
            loss = train_step(model, optimizer, images, maps)
            train.append(float(loss))
        avg_train.append(sum(train)/len(train))
        val = []
        for batch in val_ds.iter(batch_size=2):
            images = jnp.array(batch["image"])
            maps = jnp.array(batch["map"])
            loss = eval_step(model, images, maps)
            val.append(float(loss))
        avg_val.append(sum(val)/len(val))
        if best_val_loss > avg_val[-1]:
            best_val_loss = avg_val[-1]
            state = nnx.state(model)
            ocp.save(os.path.abspath(args.ckpt_dir), state, overwrite=True)
        print(f"Train Loss: {avg_train[-1]:.4f} | Val Loss: {avg_val[-1]:.4f} | Time: {(time.time()-start):.4f}s")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)