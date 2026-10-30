import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CrystalDataset, collate, split_paths
from .model import make_model, save_weights


def run_epoch(model, loader, optimizer, scaler, device, training: bool, scheduler=None):
    # TorchVision detectors return a loss dictionary only in training mode.
    model.train()
    total = 0.0
    for step, (images, targets) in enumerate(loader, 1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {losses}")
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        total += float(loss.detach()) * len(images)
        if training and step % 20 == 0:
            print(f"  batch {step}/{len(loader)} loss={float(loss.detach()):.4f}", flush=True)
    return total / len(loader.dataset)


def train(args):
    root = Path(args.data).expanduser().resolve()
    sources = tuple(args.sources.split(","))
    if any(source not in ("vmxi", "xchem") for source in sources):
        raise ValueError("--sources must contain vmxi and/or xchem")
    train_paths = split_paths(root, sources, "training")
    valid_paths = split_paths(root, sources, "validation")
    if set(train_paths) & set(valid_paths):
        raise ValueError("Training and validation images overlap")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train_loader = DataLoader(CrystalDataset(train_paths, args.max_size, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(CrystalDataset(valid_paths, args.max_size), batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda")
    model = make_model(args.coco_weights, args.max_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.3)
    history = []
    best = float("inf")
    print(f"sources={sources} train={len(train_paths)} validation={len(valid_paths)} device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, True, scheduler)
        valid_loss = run_epoch(model, valid_loader, optimizer, scaler, device, False)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss, "seconds": round(time.monotonic() - start, 2)}
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        if valid_loss < best:
            best = valid_loss
            save_weights(model, out / "best_fp16.pth", half=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)}, out / "best_training_checkpoint.pth")
    (out / "run.json").write_text(json.dumps({"sources": sources, "training_images": len(train_paths), "validation_images": len(valid_paths), "best_validation_loss": best, "args": vars(args)}, indent=2))
    return out / "best_fp16.pth"


def add_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--data", required=True, help="Unzipped Zenodo CHiMP detector dataset directory")
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--sources", default="vmxi,xchem", help="vmxi, xchem, or vmxi,xchem")
    parser.add_argument("--coco-weights", required=True, help="TorchVision Mask R-CNN v2 COCO checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
