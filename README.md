# BD PlateVision

### A YOLO11-Based Framework for Bangladeshi License Plate Localization and Character Recognition

**Detect the plate. Recognize Bengali characters. Reconstruct the
registration text.**

```{=html}
<p>
```
`<img src="https://img.shields.io/badge/Computer%20Vision-YOLO11-2563eb?style=for-the-badge" alt="Computer Vision: YOLO11">`{=html}
`<img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge" alt="FastAPI">`{=html}
`<img src="https://img.shields.io/badge/Frontend-Next.js%20%7C%20TypeScript-111827?style=for-the-badge" alt="Next.js and TypeScript">`{=html}
`<img src="https://img.shields.io/badge/Status-Research%20Project-64748b?style=for-the-badge" alt="Research project">`{=html}
```{=html}
</p>
```
**Developed by Abidul Hoque Alvee**\
Independent Computer Vision Project · 2026
:::

------------------------------------------------------------------------

## Overview

**BD PlateVision** is an end-to-end computer vision project for
detecting Bangladeshi vehicle license plates and recognizing the Bengali
text and numerals printed on them.

The machine-learning pipeline uses two YOLO11n models:

-   **License plate localization** finds the plate region in a vehicle
    image.
-   **Character recognition** detects and classifies the plate's region
    name, category letter, and digits.

A FastAPI backend exposes the recognition pipeline to a web interface
built with Next.js and TypeScript.

> **Important:** This is a research and demonstration project.
> Recognition results should be checked against the original image,
> particularly when the plate is blurred, distant, tilted, poorly
> illuminated, or when any confidence score is low.

## Table of Contents

-   [Highlights](#highlights)
-   [How It Works](#how-it-works)
-   [Technology Stack](#technology-stack)
-   [Dataset](#dataset)
-   [Models and Evaluation](#models-and-evaluation)
-   [Recent Recognition Improvements](#recent-recognition-improvements)
-   [Project Structure](#project-structure)
-   [Local Setup](#local-setup)
-   [Run the Recognition Pipeline](#run-the-recognition-pipeline)
-   [FastAPI Endpoints](#fastapi-endpoints)
-   [Google Colab Experiments](#google-colab-experiments)
-   [Limitations and Future Work](#limitations-and-future-work)
-   [Author](#author)

## Highlights

-   YOLO11-based plate localization and character-level detection.
-   Bengali region names, category letters, and digits mapped to
    readable text.
-   Two-line plate reconstruction with left-to-right character ordering.
-   Character-level confidence values and plate bounding-box
    coordinates.
-   Dataset auditing, annotation checks, and train/validation/test
    evaluation.
-   Baseline-versus-improved localization experiments.
-   FastAPI endpoint for connecting the ML pipeline to a web
    application.
-   Targeted regression tests for difficult plate examples.

## How It Works

``` text
Vehicle Image
     |
     v
YOLO11n Plate Localization
     |
     v
Plate Bounding Box
     |
     v
Padded Plate Crop
     |
     v
YOLO11n Character Detection
     |
     v
Duplicate / Overlap Filtering
     |
     v
Top and Bottom Row Reconstruction
     |
     v
Recognized Bengali Plate Text
```

The backend returns the recognized plate string, upper and lower lines,
localization confidence, plate coordinates, and individual character
detections.

## Technology Stack

  -----------------------------------------------------------------------
  Area                                Technologies
  ----------------------------------- -----------------------------------
  Detection models                    Ultralytics YOLO11n

  Deep learning                       PyTorch

  Image processing                    OpenCV, NumPy

  Backend API                         FastAPI

  Web interface                       Next.js, React, TypeScript

  Training and evaluation             Python, Google Colab, Tesla T4 GPU
                                      when available
  -----------------------------------------------------------------------

## Dataset

The project maintains separate YOLO-format datasets for plate
localization and character recognition. The original datasets were
preserved separately from cleaned or experimental copies.

### Localization dataset

The final leakage-safe localization split contains **6,405 images**.

  Split             Images   Positive images   Negative images   Plate objects
  ------------ ----------- ----------------- ----------------- ---------------
  Train              5,124             3,745             1,379           3,798
  Validation           640               446               194             450
  Test                 641               469               172             485
  **Total**      **6,405**         **4,660**         **1,745**       **4,733**

The initial audit found invalid boxes, image/XML dimension mismatches,
exact duplicates, and visually related samples across splits. These
issues were investigated and handled in separate cleaned datasets. The
final leakage-safe split had no detected cross-split MD5 duplicates,
exact perceptual-hash duplicates, or near perceptual-hash matches at the
configured threshold.

### Character-recognition dataset

  Split             Images   Character objects
  ------------ ----------- -------------------
  Train              3,480              30,604
  Validation           363               3,180
  Test                 218               1,921
  **Total**      **4,061**          **35,705**

The character detector has **27 classes**:

``` text
Digits: 0 1 2 3 4 5 6 7 8 9

Labels: ba, ca, chattogram, da, dhaka, ga, gha, ha, ja,
        jashore, ka, kha, khulna, metro, na, ta, va
```

The `data.yaml` file defines the class IDs and split paths. Keep train,
validation, and test data separated when preparing new experiments.

## Models and Evaluation

Both localization and character recognition use YOLO11n.

### Model files

``` text
models/
├── localization/
│   ├── localization_best.pt
│   └── localization_baseline_best.pt
└── character/
    └── character_best.pt
```

-   `localization_best.pt` --- selected improved localization model.
-   `localization_baseline_best.pt` --- retained baseline for
    comparison.
-   `character_best.pt` --- current production character-recognition
    model.

### Localization results

The improved localization model was selected after comparing held-out
test metrics and difficult unseen-image behavior. Validation metrics
alone did not clearly favor the improved model.

  Held-out test metric     Baseline   Improved
  ---------------------- ---------- ----------
  Precision                  0.9830     0.9814
  Recall                     0.9710     0.9794
  mAP@0.50                   0.9910     0.9926
  mAP@0.50:0.95              0.7880     0.7920

The improved model had slightly lower precision but higher recall and
mAP on the held-out test set. It also improved localization confidence
on one difficult unseen example. Some difficult examples remained
unresolved.

### Character-recognition results

Original character model on the held-out test set:

  Metric            Result
  --------------- --------
  Precision         0.9330
  Recall            0.9178
  mAP@0.50          0.9444
  mAP@0.50:0.95     0.7078

These are **object-detection metrics**, not complete license-plate
string accuracy. Results for rare classes such as `ka` and `ha` should
be interpreted cautiously because only a small number of examples are
present in the held-out splits.

## Recent Recognition Improvements

Additional testing in October 2026 investigated an unseen plate that was
sometimes classified as `ha` instead of `ka`.

### Backend adjustments tested

-   Increased the right-side plate-crop padding to 50%.
-   Enabled class-agnostic non-maximum suppression for character
    detections to reduce same-location competing predictions.
-   Used vertical character positions to separate upper and lower rows
    rather than relying only on a fixed crop-height threshold.
-   Kept inference image size at 640 because validation performance
    decreased at 960 and 1280 in the tested experiment.

These are **inference-time adjustments**, not evidence that the
character model itself was retrained successfully. The original
character checkpoint remains in use.

### Three regression images

  Test image                Expected and observed plate string     Plate confidence
  ------------------------- ------------------------------------ ------------------
  `unseen_vehicle_01.jpg`   চট্টগ্রাম মেট্রো-ক ১১-০৮৩১                          87.95%
  `images.jfif`             ঢাকা মেট্রো-গ ৩১-৯৯৫৭                             49.97%
  `images (1).jfif`         খুলনা মেট্রো-ট ১১-১২৫১                             88.11%

All three strings matched the expected strings in these targeted checks.
However, the Chattogram `ka` confidence was only 21.46%, and the Dhaka
localization confidence was 49.97%. These results are useful regression
checks on three known images; they do **not** establish general accuracy
improvement.

### Character-model experiments

Two experimental fine-tuning runs were retained:

-   `ka_balanced_t4_20261018_145008`
-   `ka_controlled_20261018_154712`

Neither experiment consistently improved the reported metrics over the
original character model. The experimental checkpoints were **not
promoted to production**. The original
`models/character/character_best.pt` was not overwritten.

The Colab notebook and experiment artifacts are retained for
reproducibility. Large checkpoints and datasets should be stored outside
GitHub unless the repository is deliberately configured for large-file
hosting.

## Project Structure

The exact tree may vary by local setup, and the web app may be in a
separate folder.

``` text
Bangladesh_License_Plate_Recognition/
├── dataset/
│   └── yolo/
│       ├── Localization/
│       └── Character Recognition/
├── final_pipeline/
│   ├── lpr_pipeline.py
│   └── pipeline_info.txt
├── inference_tests/
│   ├── local_test/
│   └── final_new_image_test/
├── models/
│   ├── localization/
│   └── character/
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_localization_improvement.ipynb
│   ├── 03_colab_training_evaluation.ipynb
│   ├── 04_final_local_inference.ipynb
│   └── BD_PlateVision_Character_Model_FineTuning.ipynb
├── results/
│   ├── character/
│   ├── final_evaluation/
│   └── localization/
├── train_character_ka_balanced.py
├── requirements.txt
├── .gitignore
└── README.md
```

Do not commit virtual environments, large dataset archives, or
experimental checkpoints unintentionally. The `.gitignore` should
exclude local-only files and generated artifacts that are not meant to
be versioned.

## Local Setup

Python **3.11 or 3.12** is recommended for a compatible local
environment. Use the version and package set supported by your installed
PyTorch and Ultralytics versions.

Open PowerShell in the project root:

``` powershell
cd "D:\Documents\Ml project\Bangladesh_License_Plate_Recognition"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already have a working virtual environment, activate it rather
than creating another one.

Check core dependencies:

``` powershell
python -c "import cv2, torch, ultralytics; print('OpenCV:', cv2.__version__); print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__)"
```

GPU training requires a compatible CUDA-enabled PyTorch installation and
an available GPU. CPU training is possible but may be much slower and
can depend on the local runtime environment.

## Run the Recognition Pipeline

The standalone ML pipeline is located at:

``` text
final_pipeline/lpr_pipeline.py
```

Example:

``` python
from final_pipeline.lpr_pipeline import BangladeshLPR

lpr = BangladeshLPR(
    "models/localization/localization_best.pt",
    "models/character/character_best.pt",
)

result = lpr.recognize("path/to/vehicle_image.jpg")

print(result["status"])
print(result.get("text"))
print(result.get("plate_box"))
print(result.get("localization_detections"))
print(result.get("characters"))
```

The returned fields depend on the pipeline response and may include
status, recognized text, plate box, localization detections, character
detections, and crop information.

## FastAPI Endpoints

The web-app backend exposes these endpoints when running locally:

  Method   Endpoint           Purpose
  -------- ------------------ ---------------------------------
  `GET`    `/`                API status
  `GET`    `/api/health`      Model and class status
  `POST`   `/api/recognize`   Upload an image for recognition

The recognition endpoint accepts JPG, PNG, and WebP images up to 10 MB.

Example health-check URL:

``` text
http://127.0.0.1:8000/api/health
```

Run the API from the directory containing its `main.py`, using the
command configured for that backend environment. A common local command
is:

``` powershell
uvicorn main:app --reload
```

Ensure that the backend model paths point to the actual `.pt` files and
that the web app's API base URL matches the running backend.

## Google Colab Experiments

The notebook `notebooks/BD_PlateVision_Character_Model_FineTuning.ipynb`
documents the character-model investigation.

The workflow includes:

1.  Mount Google Drive.
2.  Load the dataset archive and original character model.
3.  Extract the dataset and create a Colab-compatible `data.yaml`.
4.  Evaluate the original model.
5.  Run controlled augmentation and fine-tuning experiments.
6.  Compare validation and test metrics.
7.  Inspect confusing character predictions and regression images.

The notebook contains paths specific to the original Google Drive folder
structure. Update those paths when running under another account or
directory.

Experiment artifacts were saved under:

``` text
MyDrive/Bangladesh_License_Plate_Recognition/colab_artifacts/20261018/
```

This backup includes the augmented dataset, confusion-matrix artifacts,
a diagnostic crop, and an experiment summary. The experiment run folders
also contain their `best.pt` and `results.csv` files.

## Limitations and Future Work

-   Severe blur, low resolution, poor illumination, and unusual viewing
    angles remain challenging.
-   Some character classes are underrepresented and can be confused.
-   Correct predictions on a few known images do not establish broad
    real-world accuracy.
-   Confidence values are model scores, not calibrated probabilities
    that the full plate string is correct.
-   The three regression examples should be supplemented with a larger
    and more diverse evaluation set.
-   Future work should prioritize verified labels and additional diverse
    real-world examples for sparse character classes.
-   Keep the production model unchanged until a candidate demonstrates
    reliable improvement on validation, held-out test data, and
    regression images.

## Author

**Abidul Hoque Alvee**\
Independent Computer Vision Project · 2026

## Data and License Notice

Add a license before publicly redistributing the source code. Confirm
that dataset images, annotations, and model weights may be shared under
their applicable terms before publishing them.
