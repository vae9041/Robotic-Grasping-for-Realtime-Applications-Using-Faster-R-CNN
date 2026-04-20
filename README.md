# Robotic Grasping for Real-Time Applications Using Faster R-CNN

**Vincent Eloy** and **Ferat Sahin** — Rochester Institute of Technology, Rochester, NY, USA.

PyTorch codebase for **two-stage grasp detection** with Faster R-CNN (torchvision): **ResNet-50 FPN**, **ResNet-18 FPN**, and a **structured-pruned ResNet-18** variant aimed at low-latency deployment. Grasp hypotheses are **axis-aligned bounding boxes** with confidence scores; annotations from Cornell and Jacquard are converted to AABBs at **640×480**, RGB normalized to **[0, 1]**.

This repository contains **training, evaluation, offline augmentation, RealSense live inference, structured pruning, and latency/FLOP profiling scripts**. Robot-side **ROS 2**, **MoveIt 2**, **UR5e**, and **Intel RealSense D435** integration described in the paper is a separate deployment stack; here the focus is the **vision model and benchmarks**.


## Highlights (from the accompanying manuscript)

| Setting | Cornell (image-level) | Jacquard | Notes |
|--------|------------------------|----------|--------|
| ResNet-50 Faster R-CNN | **98.76%** | **90.6%** | Deepest backbone; strongest accuracy |
| Inference (representative GPU) | — | — | **~32 ms** forward time reported for the full detector (paper) |
| Structured-pruned ResNet-18 | Competitive with full R-18 (see eval logs) | — | **~7.66 ms** inference; FPN-focused pruning in-repo |

Evaluation protocol for Cornell/Jacquard in this codebase typically uses **confidence 0.5** and **IoU 0.25** for image-level success unless you override flags.

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

```bash
python -m venv faster_rcnn_env
source faster_rcnn_env/bin/activate   # Linux/macOS
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # example; pick URL for your CUDA
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Dataset layout

Download **Cornell** and **Jacquard** into the project (or symlink). Expected layout (names match defaults in `train.py`):

```
Vince_Fasterrcnn/
├── cornell_dataset/
├── augmented_dataset/          # produced by data_augmentation.py
├── jacquard_dataset/
├── train.py
└── ...
```

---

## Training

### ResNet-50 (default `train.py`)

```bash
source faster_rcnn_env/bin/activate
python train.py \
  --augmented_cornell_path ./augmented_dataset \
  --cornell_path ./cornell_dataset \
  --jacquard_path ./jacquard_dataset \
  --optimizer adamw \
  --scheduler cosine \
  --lr 0.0005 \
  --weight_decay 0.01 \
  --batch_size 8 \
  --num_epochs 30 \
  --train_ratio 0.9 \
  --output_dir ./models_adamw_cosine \
  --show_model_details
```

### ResNet-18

```bash
python train_resnet_18.py \
  --augmented_cornell_path ./augmented_dataset \
  --cornell_path ./cornell_dataset \
  --jacquard_path ./jacquard_dataset \
  --output_dir ./models_resnet18_v2
```

Use `python train.py --help` / `python train_resnet_18.py --help` for full argument lists (optimizers, schedulers, `--freeze_backbone`, etc.).

---

## Evaluation

**Cornell (ResNet-50):**

```bash
python eval_cornell.py \
  --model_path ./path/to/best_model.pth \
  --cornell_path ./cornell_dataset \
  --confidence_threshold 0.5 \
  --iou_threshold 0.25 \
  --angle_threshold_deg 30 \
  --output_dir ./evaluation_results
```

**Jacquard (ResNet-50):**

```bash
python eval_jacquard.py \
  --model_path ./path/to/best_model.pth \
  --jacquard_path ./jacquard_dataset \
  --confidence_threshold 0.5 \
  --iou_threshold 0.25 \
  --angle_threshold_deg 30 \
  --output_dir ./jacquard_evaluation
```

**ResNet-18** (same metrics; swap script):

```bash
python eval_cornell_res18.py --model_path ./path/to/resnet18.pth --cornell_path ./cornell_dataset --output_dir ./evaluation_results_res18
python eval_jacquard_res18.py --model_path ./path/to/resnet18.pth --jacquard_path ./jacquard_dataset --output_dir ./jacquard_eval_res18
```

`--angle_threshold_deg` enforces the orientation condition used in the paper:
`|theta_pred - theta_gt| <= 30°` (with 180° grasp symmetry), in addition to the IoU threshold.

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
- **Input**: RGB **640×480**, float tensor in **[0, 1]** (CHW), passed as a **list** of tensors per image.

ResNet-50 checkpoint size is on the order of **41M parameters**; ResNet-18 is smaller; structured pruning reduces parameters and estimated FLOPs (see `profile_faster_rcnn_compute.py`).

---

## Results snapshot (strong configuration)

Reported best table rows from internal evaluation logs (ResNet-50, AdamW + cosine-style training):

| Dataset | Image-level success | Notes |
|---------|---------------------|--------|
| Cornell | **98.76%** | `confidence_threshold=0.5`, `iou_threshold=0.25` |
| Jacquard | **~90.6%** | Same thresholds; full-dataset eval can be long |

Per-grasp precision/recall/F1 vary; see JSON under your chosen `--output_dir`.

---

## Troubleshooting

- **CUDA OOM**: reduce `--batch_size` in training or use a smaller backbone.
- **Dataset paths**: verify `ls cornell_dataset jacquard_dataset`.
- **Structured checkpoint load errors**: use `eval_*_res18.py` or inference scripts that call `load_structured_pruned_state_dict`; ensure `torch`/`torchvision` versions match training.

---

## Citation

If you use this code or the associated paper, please cite the manuscript (update with venue/page numbers when published):

```bibtex
@misc{eloy2026grasping,
  title={Robotic Grasping for Real-Time Applications Using Faster {R-CNN}},
  author={Eloy, Vincent and Sahin, Ferat},
  year={2026},
  institution={Rochester Institute of Technology},
  note={Draft / preprint }
}
```

---

## License

Specify your license here (e.g., MIT) if you release the repository publicly.
