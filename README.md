# Robotic Grasping for Real-Time Applications Using Faster R-CNN

**Vincent Eloy** and **Ferat Sahin** — Rochester Institute of Technology, Rochester, NY, USA.

PyTorch codebase for **two-stage grasp detection** with Faster R-CNN (torchvision): **ResNet-50 FPN**, **ResNet-18 FPN**, and a **structured-pruned ResNet-18** variant aimed at low-latency deployment. The angle branch predicts **oriented grasps** `(cx, cy, opening, jaw_size, theta)` alongside AABBs used for proposals and detection. Images are uniformly scaled and padded to **640×480**, with RGB in **[0, 1]**. The model retains Faster R-CNN’s usual internal short-side resolution.

This repository contains **training, evaluation, offline augmentation, RealSense live inference, structured pruning, and latency/FLOP profiling scripts**. Robot-side **ROS 2**, **MoveIt 2**, **UR5e**, and **Intel RealSense D435** integration described in the paper is a separate deployment stack; here the focus is the **vision model and benchmarks**.


## Highlights 

| Setting | Cornell (image-level) | Jacquard | Notes |
|--------|------------------------|----------|--------|
| ResNet-50 Faster R-CNN | **98.76%** | **90.6%** | Deepest backbone; strongest accuracy |
| Inference (NVIDIA RTX 6000 GPU) | **13.2 ms** forward time reported for the full detector |
| Structured-pruned ResNet-18 | Competitive with full R-18 (see eval logs) | **7.66 ms** inference; FPN-focused pruning in-repo |

Historical accuracy figures above describe earlier AABB-based experiments and are **not rotated-grasp results**. Corrected evaluation requires **polygon IoU > 0.25 and angle error ≤ 30°**, modulo 180°, against the same successful ground-truth grasp. By default it evaluates the top prediction at confidence 0.5 on the saved original-image holdout.

---

## Repository contents 

| Area | Role |
|------|------|
| `train.py` | Train **ResNet-50** Faster R-CNN; `GraspDataset` merges augmented Cornell + Cornell + Jacquard; default **90/10** split, seed **42**. |
| `train_resnet_18.py` | Train **ResNet-18** Faster R-CNN (same grasp formulation). |
| `data_augmentation.py` | Offline **Cornell-only** augmentation; writes `augmented_dataset/`. |
| `eval_cornell.py` / `eval_jacquard.py` | Evaluate **ResNet-50** checkpoints. |
| `eval_cornell_res18.py` / `eval_jacquard_res18.py` | Evaluate **ResNet-18** checkpoints (loads structured-pruned checkpoints when metadata is present). |
| `structured_pruning_resnet_18.py` | **Structured channel pruning** of FPN `layer_blocks` with Torch-Pruning; optional uniform FPN width via 1×1 projectors; RPN/ROI rebuild; optional fine-tune; saves `structured_pruned.pth`. |
| `inference_resnet_50.py` | Benchmark **latency** (and optional visualizations) for ResNet-50 weights. |
| `inference_resnet_18.py` | Same for **plain** ResNet-18 weights. |
| `inference_structured_pruning_resnet_18.py` | Same for **structured-pruned** checkpoints. |
| `profile_faster_rcnn_compute.py` | **Parameters + PyTorch FLOP estimate** (single forward at fixed resolution) for comparing backbones. |
| `live_inference_realsense.py` | Live **Intel RealSense** RGB grasp visualization (ResNet-50 detector in script). |
| `requirements.txt` | Python dependencies (torch/torchvision installed separately for your CUDA). |

Weights (`.pth`) are **not** tracked in git by default; see `.gitignore`. Place checkpoints locally and pass `--model_path` / `--checkpoint` as in the examples below.

---

## Environment setup

Use Python 3.10 or newer. Install a compatible torch/torchvision pair for your GPU, or their CPU builds for local testing:

```bash
python3 -m venv faster_rcnn_env
source faster_rcnn_env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Training and model smoke tests have been exercised with torch 2.14.1+cpu / torchvision 0.29.1+cpu. Torch is installed separately from `requirements.txt` so you can choose the correct CUDA build.

## Dataset layout and saved holdout

Run the commands below from this repository directory. Put or symlink extracted datasets at `cornell_dataset/` and `jacquard_dataset/`, or pass their actual roots to the scripts.

Cornell discovery uses `*cpos.txt` and matching `*r.png` files, including nested numbered folders. Neither a point-cloud text file nor depth is required for RGB training. Unsuccessful `cneg` annotations are excluded. Offline augmentation accepts depth PNG/TIFF when present, or an empty depth image when absent.

Jacquard discovery uses matching `*_grasps.txt` / `*_RGB.png` pairs. All views of the same object remain in one partition. Annotations use clockwise image-coordinate degrees, converted internally to radians; opening and jaw size retain their edge meanings.

**Create the split before augmenting:**

```bash
python prepare_split.py \
  --cornell_path ./cornell_dataset \
  --jacquard_path ./jacquard_dataset \
  --train_ratio 0.9 \
  --output ./dataset_split.json

python data_augmentation.py \
  --cornell_path ./cornell_dataset \
  --split_manifest ./dataset_split.json \
  --output_path ./augmented_dataset_v2 \
  --num_augmentations 5
```

Only training identities are augmented. The training loader also enforces this restriction, even if an augmentation directory contains validation-derived images. Old augmentations lack corrected geometry/provenance and must be regenerated into a new directory. Validation always uses original images and all their valid positive grasps, without the training-only grasp-count cap.

If you omit `--split_manifest` in training, the script creates and saves a grouped split from original images, or reuses the existing output directory’s manifest. The manifest is also embedded in each checkpoint. Cornell is split by original image; Jacquard by object. Splits are deterministic and stratified by dataset, so the image ratio for Jacquard can differ slightly from the requested group ratio.

## Training

### ResNet-50

```bash
python train.py \
  --cornell_path ./cornell_dataset \
  --jacquard_path ./jacquard_dataset \
  --augmented_cornell_path ./augmented_dataset_v2 \
  --split_manifest ./dataset_split.json \
  --optimizer adamw --scheduler cosine \
  --lr 0.0005 --weight_decay 0.01 \
  --batch_size 4 --num_epochs 30 --device cuda \
  --output_dir ./models_resnet50_oriented
```

### ResNet-18

```bash
python train_resnet_18.py \
  --cornell_path ./cornell_dataset \
  --jacquard_path ./jacquard_dataset \
  --augmented_cornell_path ./augmented_dataset_v2 \
  --split_manifest ./dataset_split.json \
  --batch_size 4 --num_epochs 50 --device cuda \
  --output_dir ./models_resnet18_v2
```

Use `--device cpu` for CPU execution. Both backbones share the oriented geometry head, pi-periodic angle loss, preprocessing, and polygon evaluator. Checkpoints include geometry-version and split metadata. `final_model.pth`, best checkpoints, `dataset_split.json`, and `training_curves.png` are written to the output directory.

### Warm-start an older ResNet-50 detector

Old AABB-only and pre-fix angle checkpoints cannot directly provide corrected rotated grasps. Their detector weights can initialize training, while the angle and dimension heads are reset:

```bash
python train.py \
  --resume /path/to/old_resnet50_checkpoint.pth \
  --finetune_angle_head \
  --cornell_path ./cornell_dataset \
  --jacquard_path ./jacquard_dataset \
  --augmented_cornell_path ./augmented_dataset_v2 \
  --split_manifest ./dataset_split.json \
  --optimizer adamw --lr 0.0001 --scheduler cosine \
  --num_epochs 10 --device cuda \
  --output_dir ./models_resnet50_grasp_heads
```

`--finetune_angle_head` trains both angle and actual grasp-dimension heads while preserving detector weights. A later run with `--resume ... --unfreeze_all` starts joint fine-tuning with a fresh optimizer. Resume without a fine-tuning flag requires a corrected checkpoint and restores the optimizer, scheduler, and original split.

An older checkpoint may already have seen images in a newly created holdout. Such warm starts are automatically marked as **unverified legacy pretraining** in evaluation output. To measure an unbiased held-out result, train from generic pretrained weights using the saved split, or establish an independently untouched evaluation set. Historical AABB accuracy does not establish rotated-grasp accuracy.

## Evaluation

The evaluators use the checkpoint’s saved validation identities automatically. An explicit manifest must match the embedded training split. They report AABB success, angle success conditional on AABB localization, and true rotated-grasp success separately.

```bash
python eval_cornell.py \
  --model_path ./models_resnet50_oriented/best_model.pth \
  --cornell_path ./cornell_dataset \
  --confidence_threshold 0.5 --iou_threshold 0.25 --angle_threshold 30 \
  --output_dir ./cornell_oriented_evaluation --visualize

python eval_jacquard.py \
  --model_path ./models_resnet50_oriented/best_model.pth \
  --jacquard_path ./jacquard_dataset \
  --confidence_threshold 0.5 --iou_threshold 0.25 --angle_threshold 30 \
  --output_dir ./jacquard_oriented_evaluation
```

For ResNet-18, use `eval_cornell_res18.py` / `eval_jacquard_res18.py`; they use the same evaluator and support corrected structured-pruned checkpoints. `--top_k` defaults to 1. Raising it changes the evaluation protocol. `--allow_full_dataset` is an explicit diagnostic mode that labels results as including training images, not a holdout score.

Metrics are saved to `evaluation_results.json`; visualization draws actual rotated polygons. An angle and IoU must pass for the **same** ground-truth annotation, and any qualifying annotation can match. Per-grasp matching is one-to-one and ordered by prediction confidence.

## Verification

```bash
python -m unittest discover -s tests -v
```

The suite uses synthetic annotations and real CPU forward/backward passes for both backbones. It requires no datasets, camera, CUDA, or pretrained downloads. It checks corner and angle conventions, polygon matching, augmentation alignment, negative filtering, grouped holdouts, geometry encoding, transforms, checkpoint migration, and evaluator selection. Dataset-level accuracy still requires actual training and evaluation.

---

## Inference timing (latency benchmarks)

Single-GPU timings depend on device and precision; scripts synchronize CUDA and average multiple runs:

```bash
python inference_resnet_50.py       --checkpoint ./path/to/final_model.pth --image_dir ./images --fp16
python inference_resnet_18.py       --checkpoint ./path/to/final_model.pth --image_dir ./images --fp16
python inference_structured_pruning_resnet_18.py --checkpoint ./path/to/structured_pruned.pth --image_dir ./images --fp16
```

---

## Structured pruning (ResNet-18)

Pruning is implemented in **`structured_pruning_resnet_18.py`**: Torch-Pruning **magnitude** structured pruning on **FPN 3×3 `layer_blocks`**, optional **uniform channel width** across pyramid levels, **RPN/ROI head rebuild**, and optional **fine-tuning**. Example:

```bash
python structured_pruning_resnet_18.py \
  --checkpoint ./models_resnet18_v2/final_model.pth \
  --output_dir ./models_resnet18_v3_structured \
  --pruning_ratio 0.35 \
  --prune_iterations 8 \
  --example_height 480 --example_width 640 \
  --finetune_epochs 5 --lr 1e-4
```

Load pruned weights for eval or inference using the same scripts as ResNet-18; evaluators call `load_structured_pruned_state_dict` when `structured_pruning_meta` is in the checkpoint.

---

## Model complexity (FLOPs / parameters)

```bash
python profile_faster_rcnn_compute.py \
  --resnet50_ckpt ./path/to/resnet50.pth \
  --resnet18_ckpt ./path/to/resnet18.pth \
  --pruned_ckpt ./path/to/structured_pruned.pth
```

---

## Live RealSense inference

```bash
python live_inference_realsense.py \
  --model_path ./path/to/best_model.pth \
  --confidence_threshold 0.5 \
  --device auto
```

Requires `pyrealsense2` and a connected RealSense camera.

---

## Model architecture summary

- **Detector**: torchvision **Faster R-CNN** with **ResNet-50 + FPN** or **ResNet-18 + FPN**.
- **Head**: `FastRCNNPredictor` with **2 classes** (background, grasp).
- **Input**: aspect-preserving RGB **640×480** canvas in **[0, 1]** (CHW), passed as a **list** of tensors.
- **Outputs**: AABBs for detection plus `grasps` of shape `(N, 5)` containing `(cx, cy, opening, jaw_size, theta_radians)` in the caller image’s coordinates. Angles are clockwise in image coordinates and periodic modulo pi.
- **Losses**: classification, AABB regression, RPN losses, actual grasp geometry regression, and pi-periodic angle loss.

ResNet-50 checkpoint size is on the order of **41M parameters**; ResNet-18 is smaller; structured pruning reduces parameters and estimated FLOPs (see `profile_faster_rcnn_compute.py`).

---

## Results 

Historical AABB-based results from internal evaluation logs (ResNet-50, AdamW + cosine-style training); these do not measure the corrected rotated-grasp criterion:

| Dataset | Image-level success | Notes |
|---------|---------------------|--------|
| Cornell | **98.76%** | `confidence_threshold=0.5`, `iou_threshold=0.25` |
| Jacquard | **~90.6%** | Same thresholds; full-dataset eval can be long |

Per-grasp precision/recall/F1 vary; see JSON under your chosen `--output_dir`.

---
## Trained Models

Trained models are in: model_resnet_50/, models_resnet18_v2/, Pruned_model_v3/ accessed here: https://drive.google.com/drive/folders/17vggB7ATP2DyJ13ozPqiVgVsqpdW-pKY?usp=sharing

---
## Troubleshooting

- **CUDA OOM**: reduce `--batch_size` in training or use a smaller backbone.
- **Dataset paths**: verify `ls cornell_dataset jacquard_dataset`.
- **Structured checkpoint load errors**: use `eval_*_res18.py` or inference scripts that call `load_structured_pruned_state_dict`; ensure `torch`/`torchvision` versions match training.

---
