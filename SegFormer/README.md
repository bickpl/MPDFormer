# SegFormer Training

This folder trains SegFormer on CRACK500 using a training recipe closer to the public SegFormer configuration instead of the old fixed 256 tiled-patch loader.

## Main Changes

- MiT-B0 to MiT-B5 are supported.
- MiT HuggingFace weights are converted to the local MixVisionTransformer key format before loading.
- The loader prints the number of backbone keys loaded, for example:
  - `Loaded MiT-b5 pretrained backbone keys: 948/948`
- `--load` is now only for a full model/Lightning checkpoint, not for MiT backbone weights.
- Training uses random resize ratio + random crop:
  - default crop size: `640`
  - default ratio range: `0.5` to `2.0`
  - horizontal flip
  - simple photometric distortion
  - ImageNet mean/std normalization
- Validation uses resized full images padded to a multiple of 32.
- Optimizer is AdamW with poly warmup scheduling.
- Decode head uses `10x` learning rate by default.

## B5 Training

```powershell
D:\anaconda3\envs\torch1.8\python.exe train_code\SegFormer\train.py `
  --variant b5 `
  --crop-size 640 `
  --scale 1.0 `
  --ratio-min 0.5 `
  --ratio-max 2.0 `
  --batch-size 2 `
  --learning-rate 6e-5 `
  --head-lr-mult 10 `
  --weight-decay 0.01 `
  --warmup-iters 1500 `
  --amp
```

If memory is tight, keep the public crop recipe but reduce batch pressure:

```powershell
D:\anaconda3\envs\torch1.8\python.exe train_code\SegFormer\train.py `
  --variant b5 `
  --crop-size 640 `
  --batch-size 1 `
  --accumulate-grad-batches 2 `
  --amp
```

## Download Weights

```powershell
D:\anaconda3\envs\torch1.8\python.exe train_code\SegFormer\download_pretrained.py --variant b5
```
