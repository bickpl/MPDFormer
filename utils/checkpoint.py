from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, best=None, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "config": config or {},
        "best": best or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=False):
    checkpoint = torch.load(str(path), map_location=map_location)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint, list(missing), list(unexpected)

