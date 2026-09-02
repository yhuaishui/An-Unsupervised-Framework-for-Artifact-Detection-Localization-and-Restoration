# Histopathology Artifact Detection, Localization, and Restoration

The official code of "Toward Robust Histopathology Imaging: An Unsupervised Framework for Artifact Detection, Localization, and Restoration"

This folder contains the implementation of an automatic pipeline for **artifact detection, localization, and restoration in histopathology (H&E) patch images**. Given an input patch, the pipeline:

1. **Detects** whether the patch contains artifacts (e.g., pen marks, bubbles, folding, tissue tearing) using a FastFlow anomaly detection model;
2. **Localizes** the artifact regions by combining the anomaly heatmap with unsupervised image segmentation and post-processing;
3. **Restores** the artifact regions with a conditional denoising diffusion model (Palette-style inpainting) that re-generates the missing tissue content.

An interactive demo is provided in [`demo_pipeline.ipynb`](demo_pipeline.ipynb) with sample patches under [`data/sample_patch`](data/sample_patch).

## 📦 To-Do List

1. **Simulation of artifact datasets** Generate synthetic artifacts to enable robust, unsupervised validation of model performance and to derive standardized parameters for heatmap generation.
2. **Whole-slide image (WSI) inference** Implement efficient inference pipelines capable of processing large-scale histopathological slides.
3. **Pretrained weights release** Provide publicly accessible model checkpoints to facilitate reproducible research and downstream fine-tuning across diverse applications.

## Directory Structure

```
code/
├── his_pipeline.py                  # Main entry: Pathological pipeline (detect → locate → restore)
├── demo_pipeline.ipynb              # Interactive demo on sample patches
├── train_fastflow.py                # Train the FastFlow artifact detector
├── train_restoration.py             # Train the diffusion-based restoration model
├── config/                          # JSON configs for training and inference
├── core/                            # Shared infrastructure: config parsing, logging, base classes, utils
├── data/                            # Datasets
├── models/                          # FastFlow, diffusion U-Net, Palette model, losses, metrics
└── pipeline/                        # Detection/localization and restoration inference modules
```

## Requirements

The code is implemented and tested with Python 3.8 and PyTorch 1.13.1. To install the dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

Run the full pipeline on a sample patch:

```python
from his_pipeline import Pathological

pipeline = Pathological(config_path='config/config_pipeline.json')
result = pipeline.pipeline(image_path='data/sample_patch/artifact_1.png')

# result keys: image, anomaly_map, pred_label, mask, restore
print('Detected as artifact:', result['pred_label'])
```

The returned dict contains:

| Key           | Description                                                     |
|---------------|-----------------------------------------------------------------|
| `image`       | Input image resized to the model input size                     |
| `anomaly_map` | Normalized anomaly heatmap in `[0, 1]`                 |
| `pred_label`  | Image-level prediction: whether the patch contains artifacts    |
| `mask`        | Pixel-level artifact localization mask                          |
| `restore`     | Restored image (original image if no artifact is detected)      |

### Pipeline API

The `Pathological` class in [`his_pipeline.py`](his_pipeline.py) exposes the pipeline as individual stages:

| Method      | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `pipeline()`| End-to-end detect → locate → restore                               |
| `detect()`  | Artifact detection only: returns image, anomaly map, pred label    |
| `locate()`  | Artifact localization only: returns image, anomaly map, mask       |
| `restore()` | Restoration only: inpaints the image given a mask                  |

Each method accepts either an `image_path` or an in-memory image (`PIL.Image` or NumPy array).

## Training

### FastFlow (artifact detection and localization)

Train the FastFlow anomaly detector with [`train_fastflow.py`](train_fastflow.py):

```bash
python train_fastflow.py -c config/config_fastflow.json -p train
```

Key settings in [`config/config_fastflow.json`](config/config_fastflow.json):
- `datasets.train.dataset.data_root`: directory of normal training patches;
- `datasets.val.dataset.imgs_root` / `gts_root`: validation images and ground-truth masks;
- `model.backbone_name`: frozen feature backbone (default: `prov-gigapath` from Hugging Face, downloaded on first use);
- `path.resume_path` / `path.resume_state`: continue training from a checkpoint;
- `train.num_epochs`, `train.eval_interval`, `train.checkpoint_interval`: training schedule.

During training, pixel-level and image-level AUROC are computed on the validation set, and checkpoints plus an AUROC curve are saved under the experiment directory.

To evaluate a trained checkpoint:

```bash
python train_fastflow.py -c config/config_fastflow.json -p test -eval config/config_fastflow_evaluate.json
```

Set `evaluate.resume_path` to the checkpoint in `config_fastflow.json`, and the evaluation result (pixel AUROC, best threshold, min/max of the anomaly map) is written to the file given by `-eval`.

### Restoration (diffusion inpainting)

Train the diffusion restoration model with [`train_restoration.py`](train_restoration.py):

```bash
python train_restoration.py -c config/config_restoration_model.json -p train
```

Key settings in [`config/config_restoration_model.json`](config/config_restoration_model.json):
- `datasets.train.which_dataset.args.data_root`: directory of training patches;
- `datasets.train.which_dataset.args.mask_config.mask_mode`: mask generation strategy during training (e.g., `hybrid`);
- `model.which_networks[0].args.unet`: U-Net architecture of the denoiser;
- `model.which_networks[0].args.beta_schedule`: noise schedule for train/test phases;
- `path.resume_path` / `path.resume_state`: resume model weights / full training state.

Single-GPU training runs directly; for multi-GPU training the script spawns one process per GPU (DistributedDataParallel) and uses the port given by `-P`.

For more details, please refer to the original Palette GitHub repository: [`Palette-Image-to-Image-Diffusion-Models`](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

## Inference Configuration

The top-level [`config/config_pipeline.json`](config/config_pipeline.json) wires the two stages together:

```json
{
  "pipeline": {
    "detection_and_localization": {
      "config_path": "config/config_detection_localization.json"
    },
    "restoration_model": {
      "config_path": "config/config_restoration_model.json",
      "model_path": "checkpoint/restore.pth"
    }
  }
}
```

- `pipeline.detection_and_localization.config_path`: points to the detection/localization config;
- `pipeline.restoration_model.model_path`: path to the trained restoration checkpoint.

[`config/config_detection_localization.json`](config/config_detection_localization.json) controls the detector and locator:

| Section       | Key                            | Description                                                    |
|---------------|--------------------------------|----------------------------------------------------------------|
| `fastflow`    | `backbone_name`                | Feature backbone (timm `hf_hub` id)                            |
| `fastflow`    | `resume_and_normalize_config_path` | Config with checkpoint path and anomaly-map normalization stats |
| `detection`   | `abnormal_pixel_percentage`    | Minimum abnormal-pixel ratio for an image-level positive       |
| `background`  | `remove_background` / HSV ranges | Whether and how to exclude slide background                    |
| `localization`| `model`, `felzenszwalb`, `number_seg_runs`, ... | Unsupervised segmentation and mask refinement parameters |

The normalization stats and detector checkpoint are stored in [`config/config_fastflow_evaluate.json`](config/config_fastflow_evaluate.json) (or `config_fastflow_evaluate_tcga.json`): Update `resume_path` to the actual FastFlow checkpoint before running inference.

## Pretrained Models

Pretrained weights will be uploaded to [`Hugging Face`](https://huggingface.co/yhuaishui/Histopathology-Artifact-Pipeline) shortly. If you have recommendations for other cloud storage platforms, please feel free to let us know.

Before running inference, place your trained weights at the paths referenced by the configs:

- FastFlow detector: `checkpoint/fastflow.pt` (or `checkpoint/fastflow_tcga.pt`) via `config_fastflow_evaluate*.json`;
- Restoration model: `checkpoint/restore.pth` via `config_pipeline.json`.

The feature backbone (`prov-gigapath`) is fetched from Hugging Face Hub automatically on first use, which requires network access.

## Notes

- For any feedback or inquiries, please contact yanghuaishui@foxmail.com
