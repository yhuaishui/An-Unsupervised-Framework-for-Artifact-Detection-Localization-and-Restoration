# Histopathology Artifact Detection, Localization, and Restoration

The official code of "Toward Robust Histopathology Imaging: An Unsupervised Framework for Artifact Detection, Localization, and Restoration"

This folder contains the implementation of an automatic pipeline for **artifact detection, localization, and restoration in histopathology (H&E) patch images**. Given an input patch, the pipeline:

1. **Detects** whether the patch contains artifacts (e.g., pen marks, bubbles, folding, tissue tearing) using a FastFlow anomaly detection model;
2. **Localizes** the artifact regions by combining the anomaly heatmap with unsupervised image segmentation and post-processing;
3. **Restores** the artifact regions with a conditional denoising diffusion model (Palette-style inpainting) that re-generates the missing tissue content.

An interactive demo is provided in [`demo_pipeline.ipynb`](demo_pipeline.ipynb) with sample patches under [`data/sample_patch`](data/sample_patch). The same notebook also has a **WSI ROI** section that visualizes whole-slide inference results, using the sample region of interest shipped in `sample_wsi_roi` ([`can be downloaded from Hugging Face`](https://huggingface.co/yhuaishui/Histopathology-Artifact-Pipeline)).
<p align="center">
<img src=data/fig1.jpg />
</p>

## Directory Structure

```
code/
├── his_pipeline.py                  # Main entry: Pathological pipeline (detect → locate → restore)
├── demo_pipeline.ipynb              # Interactive demo on sample patches and a sample WSI ROI
├── train_fastflow.py                # Train the FastFlow artifact detector
├── train_restoration.py             # Train the diffusion-based restoration model
├── run_wsi_detect_locate.py         # WSI stage 1: patch-wise detection/localization → mask.tiff, map.tiff, detail_result.csv
├── run_wsi_restore.py               # WSI stage 2: patch-wise restoration → restore.tiff
├── config/                          # JSON configs for training and inference
├── core/                            # Shared infrastructure: config parsing, logging, base classes, utils
├── data/                            # Datasets
│   ├── sample_patch/                # Sample patches for the patch-level demo
│   └── sample_wsi_roi/              # Sample WSI ROI: img.tiff, coords.h5, mask.tiff, map.tiff, detail_result.csv, restore.tiff
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

## Whole-Slide Image (WSI) Inference

The patch-level pipeline is extended to whole-slide images (or to a WSI region of interest) by tiling the slide into `512 × 512` patches at level 0 (subsequently resized to `256 × 256`), running the pipeline patch by patch, and stitching the patch results back into pyramidal TIFFs. Two standalone scripts implement the two stages; both are launched from the repository root and reuse the configs of the patch-level pipeline.

### Stage 1 — Detection and localization (`run_wsi_detect_locate.py`)

```bash
python run_wsi_detect_locate.py
```

The paths are defined at the top of the `__main__` block:

| Variable            | Default                                  | Description                                                                    |
|---------------------|------------------------------------------|--------------------------------------------------------------------------------|
| `WSI_PATH`          | `./data/sample_wsi_roi/img.tiff`         | Slide or ROI to process (read with OpenSlide)                                  |
| `coord_path`        | `./data/sample_wsi_roi/coords.h5`        | HDF5 file holding a `coords` dataset with the level-0 `(x, y)` patch origins (can be generated by [`TRIDENT`](https://github.com/mahmoodlab/TRIDENT/) or [`CLAM`](https://github.com/mahmoodlab/CLAM))   |
| `MASK_OUTPUT_PATH`  | `./data/sample_wsi_roi/mask.tiff`        | Output artifact mask for the whole ROI                                         |
| `MAP_OUTPUT_PATH`   | `./data/sample_wsi_roi/map.tiff`         | Output anomaly map for the whole ROI                                           |
| `CSV_OUTPUT_PATH`   | `./data/sample_wsi_roi/detail_result.csv`| Output per-patch results                                                       |
| `PATCH_SIZE`        | `(512, 512)`                             | Patch size, read at level 0                                                    |
| `config_path`       | `config/config_detection_localization.json` | Detector/locator config                                                     |

Every patch listed in `coords.h5` is read from the slide, passed to `HisAnomalyModel.detection_and_localization(..., detail=True)`, and the resulting mask and anomaly map are written into shared level-0 canvases. 

`detail_result.csv` reports one row per patch, in the same order as `coords.h5`:

| Column                    | Description                                                              |
|---------------------------|--------------------------------------------------------------------------|
| `x`, `y`                  | Patch origin at level 0                                                  |
| `artifact_label`          | Image-level artifact prediction from the anomaly map                     |
| `background_score`        | Overlap between the anomaly map and the slide background                 |
| `background_percentage`   | Background pixel ratio inside the patch                                  |
| `is_background`           | Whether the patch is discarded as background                             |
| `refined_artifact_label`  | Final artifact label after background removal and mask cleaning          |
| `mask_percentage`         | Artifact-mask pixel ratio inside the patch                               |

### Stage 2 — Restoration (`run_wsi_restore.py`)

```bash
python run_wsi_restore.pyw
```

Stage 2 consumes the outputs of stage 1 (`detail_result.csv`, `img.tiff`, `mask.tiff`) and writes `restore.tiff`. Patches are handled differently depending on their labels:

- `refined_artifact_label == False` → the patch is copied unchanged into the output;
- `refined_artifact_label == True` and `mask_percentage < 0.7` → the patch is restored with the diffusion model (`RestorationModel.inpaint`);
- `refined_artifact_label == True` and `mask_percentage >= 0.7` (an excessively large mask) → the patch is currently skipped, so it stays at the white initialization of the output canvas.

The restored canvas is initialized to white, so patches that are never written keep a blank background.

### Parallelism and multi-GPU inference

Both scripts expose `num_processes` and `gpu_ids` in `main()` (defaults: `num_processes=1`, `gpu_ids=[0]`). To use several GPUs, set `num_processes` to the number of GPUs and pass the device list in `gpu_ids`.

### Sample WSI ROI and demo

The `# wsi roi sample` section of [`demo_pipeline.ipynb`](demo_pipeline.ipynb) loads the four TIFFs of the sample ROI with OpenSlide and shows, at a thumbnail resolution, the input image, the anomaly map, the artifact mask and the restored image side by side. It then draws the patch grid read from `coords.h5` on the input image, together with an overlay of the predicted artifact mask.

The sample under `sample_wsi_roi` ([`can be downloaded from Hugging Face`](https://huggingface.co/yhuaishui/Histopathology-Artifact-Pipeline)) is a 5 × 5 patch ROI (`2560 × 2560` pixels at level 0) and contains:

| File                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `img.tiff`          | Sample WSI ROI (input image)                                                |
| `coords.h5`         | `coords` dataset with the level-0 `(x, y)` origins of the 25 patches        |
| `mask.tiff`         | Artifact mask produced by stage 1                                           |
| `map.tiff`          | Anomaly map produced by stage 1                                             |
| `detail_result.csv` | Per-patch results produced by stage 1                                       |
| `restore.tiff`      | Restored ROI produced by stage 2                                            |


## Training

### Simulation of artifact datasets (used for validating model performance and deriving standardized parameters for heatmap generation)

To simulate artifact data, you may refer to the GitHub repository: [`robustness_benchmark`](https://github.com/superjamessyx/robustness_benchmark) and [`FrOoDo`](https://github.com/MECLabTUDA/FrOoDo)

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

Pretrained weights can be downloaded from [`Hugging Face (yhuaishui/Histopathology-Artifact-Pipeline)`](https://huggingface.co/yhuaishui/Histopathology-Artifact-Pipeline). If you have recommendations for other cloud storage platforms, please feel free to let us know.

Before running inference, place your trained weights at the paths referenced by the configs:

- FastFlow detector: `checkpoint/fastflow.pt` (or `checkpoint/fastflow_tcga.pt`) via `config_fastflow_evaluate*.json`;
- Restoration model: `checkpoint/restore.pth` via `config_pipeline.json`.

The feature backbone (`prov-gigapath`) is fetched from Hugging Face Hub automatically on first use, which requires network access.

## Notes

- For any feedback or inquiries, please contact yanghuaishui@foxmail.com
