# Bangladesh License Plate Recognition

An end-to-end computer vision system for detecting and recognizing
Bangladeshi vehicle license plates from vehicle images.

The system uses two YOLO11n models:

1.  **License Plate Localization** - detects the license plate region
    from a vehicle image.
2.  **Character Recognition** - detects and classifies Bangla plate
    components and digits from the localized plate.

The final pipeline performs:

**Vehicle Image -\> Plate Localization -\> Plate Crop -\> Character
Detection -\> Duplicate Filtering -\> Row Reconstruction -\> Recognized
Plate Text**

---

## 1. Project Status

The complete machine-learning pipeline has been trained, evaluated,
improved, selected, and verified in both Google Colab and a local
Windows/VS Code environment.

Current status:

- Dataset audit and cleaning: **Completed**
- Data leakage analysis: **Completed**
- Leakage-safe localization dataset: **Completed**
- YOLO dataset conversion: **Completed**
- Baseline localization training: **Completed**
- Character recognition training: **Completed**
- Held-out model evaluation: **Completed**
- End-to-end recognition pipeline: **Completed**
- Unseen-image testing: **Completed**
- Localization dataset diagnosis: **Completed**
- Targeted localization dataset improvement: **Completed**
- Improved localization retraining: **Completed**
- Baseline vs improved comparison: **Completed**
- Difficult unseen-image re-testing: **Completed**
- Final localization model selection: **Completed**
- Final end-to-end verification: **Completed**
- Local VS Code verification with final models: **Completed**

The improved YOLO11n localization model is the selected final
localization model. The original baseline localization model is retained
separately for comparison and reproducibility.

---

## 2. Dataset Preparation

Two main datasets were used:

- License Plate Localization
- Character Recognition

The original datasets were preserved without modification.

### Localization Dataset Audit

Original localization dataset:

Split Images

---

Train 5,687
Validation 277
Test 553
**Total** **6,517**

The audit identified:

- 1,795 empty annotations
- 6 invalid bounding boxes
- 218 image/XML dimension mismatches
- Cross-split duplicate and visual-overlap issues

Empty annotations were retained because they provide useful
negative/background examples.

Annotation dimension mismatches and invalid bounding boxes were
corrected in a separate cleaned dataset while preserving the original
data.

---

## 3. Data Leakage Prevention

A detailed leakage audit was performed before final training.

The localization dataset initially contained:

- 33 exact MD5 cross-split duplicate groups
- 68 files involved in exact duplication

After exact duplicate removal:

- 6,405 unique images remained
- 4,660 positive images
- 1,745 negative images
- 4,733 plate objects

Perceptual hashing and source-aware grouping were then used to prevent
visually related samples from appearing across different splits.

### Final Leakage-Safe Localization Split

Split Images Positive Negative Objects

---

Train 5,124 3,745 1,379 3,798
Validation 640 446 194 450
Test 641 469 172 485

Final verification found:

- MD5 cross-split duplicates: **0**
- Exact perceptual-hash cross-split duplicates: **0**
- Near perceptual-hash cross-split matches at threshold \<=4: **0**

This leakage-safe split was used for the baseline localization
experiment. The improved experiment preserved the same validation and
test sets and modified only the training split through targeted
augmentation.

---

## 4. Character Recognition Dataset

Character recognition dataset:

Split Images

---

Train 3,480
Validation 363
Test 218
**Total** **4,061**

Total annotated character objects:

**35,705**

The character dataset contains **27 classes**:

```text
0 1 2 3 4 5 6 7 8 9
ba ca chattogram da dhaka ga gha ha ja jashore
ka kha khulna metro na ta va
```

No cross-split MD5 or source-name leakage was detected.

---

## 5. YOLO Training Datasets

### Baseline Localization Dataset

Split Images Objects Empty Labels

---

Train 5,124 3,798 1,379
Validation 640 450 194
Test 641 485 172

### Improved Localization Dataset

The improved experiment added 536 targeted training images while leaving
validation and test unchanged.

Split Images

---

Train 5,660
Validation 640
Test 641

### Character Recognition Dataset

Split Images Objects

---

Train 3,480 30,604
Validation 363 3,180
Test 218 1,921

No missing or invalid YOLO labels were detected in the final converted
datasets.

---

## 6. Models

Both tasks use **YOLO11n**.

Final model files:

```text
models/
├── localization/
│   ├── localization_best.pt
│   └── localization_baseline_best.pt
└── character/
    └── character_best.pt
```

- `localization_best.pt` - selected improved localization model.
- `localization_baseline_best.pt` - preserved baseline localization
  model.
- `character_best.pt` - final character recognition model.

---

## 7. Baseline Localization Results

The baseline localization model was trained for a maximum of 50 epochs
with early stopping.

Training stopped at epoch 44, with the best model obtained at
approximately epoch 34.

### Validation Performance

Metric Score

---

Precision 0.972
Recall 0.969
mAP@0.50 0.993
mAP@0.50:0.95 0.802

### Held-Out Test Performance

Test set:

- 641 images
- 485 annotated plate instances
- 172 background/negative images

Metric Score

---

Precision **0.983**
Recall **0.971**
mAP@0.50 **0.991**
mAP@0.50:0.95 **0.788**

Baseline evaluation artifacts are stored in:

```text
results/localization/localization_test/
```

---

## 8. Character Recognition Results

Held-out character test set:

- 218 images
- 1,921 character instances
- 27 classes

### Aggregate Test Performance

Metric Score

---

Precision **0.933**
Recall **0.918**
mAP@0.50 **0.944**
mAP@0.50:0.95 **0.708**

Some sparse classes have fewer training/test examples and therefore
remain more difficult than frequently represented classes.

Evaluation plots are stored in:

```text
results/character/character_test/
```

---

## 9. End-to-End Recognition Pipeline

The final inference implementation is:

```text
final_pipeline/lpr_pipeline.py
```

The pipeline performs:

1.  Load vehicle image
2.  Detect license plate candidates
3.  Merge overlapping/nearby localization boxes
4.  Select the primary plate region
5.  Apply horizontal crop padding
6.  Detect plate characters
7.  Remove spatial duplicate/conflicting character detections
8.  Separate plate text into rows
9.  Sort characters left-to-right
10. Reconstruct the final plate text

### Current Inference Settings

```text
Localization confidence: 0.05
Character confidence:    0.10
Localization image size: 640
Character image size:    640
Horizontal plate padding: 12%
```

Additional pipeline information is available in:

```text
final_pipeline/pipeline_info.txt
```

---

## 10. Initial Unseen-Image Testing

Testing on previously unseen images revealed an important difference
between benchmark performance and real-world generalization.

Clear, front-facing plates were generally recognized successfully.

More difficult cases included:

- dark plates
- night/low-light conditions
- blurry images
- very low-resolution plates
- partially localized plates
- unusual viewpoints
- localization boxes cutting off part of the plate

The experiments showed that **license plate localization was the main
bottleneck**.

In several difficult examples, character recognition performed well when
a complete plate crop was manually provided, while the baseline
localization model failed to produce a complete plate region.

This motivated the localization dataset diagnosis and improvement
stages.

---

## 11. Localization Dataset Diagnosis

Analysis of the localization training data showed:

### Plate Brightness

From 3,798 training plate crops:

- Dark: 568 (14.96%)
- Medium brightness: 2,970 (78.20%)
- Bright: 260 (6.85%)

### Plate Pixel Size

Median plate size:

- Width: 148 px
- Height: 88.5 px

Small plates:

- Width \<100 px: 619
- Height \<50 px: 204
- Both width \<100 and height \<50: 186

The dataset contains moderate vehicle and background diversity, but
difficult conditions such as low resolution, poor lighting, unusual
viewpoints, and distant plates are comparatively underrepresented.

Detailed diagnosis:

```text
results/localization_dataset_diagnosis.txt
```

---

## 12. Localization Dataset Improvement

A targeted augmentation strategy was used instead of simply repeating
the baseline training experiment.

A total of **268 unique difficult source images** were selected.

### Low-Light Augmentation

- Brightness factor: 0.55
- 268 new images

### Low-Resolution Simulation

Images were downsampled to approximately one-third resolution and then
upsampled to the original dimensions to simulate low-resolution plate
capture.

- 268 new images

### Total Added Training Samples

**536 targeted augmented images**

Validation and test sets were intentionally left unchanged so that the
improved model could be compared fairly against the baseline.

### Improved Dataset

Split Images

---

Train 5,660
Validation 640
Test 641

The improved dataset passed label-integrity verification before
training.

---

## 13. Improved Localization Experiment

The improved YOLO11n localization model was trained for 50 epochs using
the targeted augmented training dataset.

The best weights are retained as the final localization model.

### Improved Validation Performance

Metric Score

---

Precision 0.973
Recall 0.965
mAP@0.50 0.993
mAP@0.50:0.95 0.797

### Improved Held-Out Test Performance

The unchanged held-out test set contained 641 images and 485 annotated
plate instances.

Metric Score

---

Precision **0.9814**
Recall **0.9794**
mAP@0.50 **0.9926**
mAP@0.50:0.95 **0.7920**

Improved evaluation artifacts are stored in:

```text
results/localization/localization_improved_test/
```

---

## 14. Baseline vs Improved Localization

Because validation performance alone did not clearly favor the improved
model, final selection was based on the unchanged held-out test set
together with difficult unseen-image behavior.

### Held-Out Test Comparison

Metric Baseline Improved Result

---

Precision **0.9830** 0.9814 Baseline slightly higher
Recall 0.9710 **0.9794** Improved
mAP@0.50 0.9910 **0.9926** Improved
mAP@0.50:0.95 0.7880 **0.7920** Improved

The precision reduction is small, while recall and both mAP measures
improved on the held-out test set.

### Difficult Unseen Cases

**Test 03**

- Baseline: no localization detection
- Improved: no localization detection
- Result: unresolved

**Test 04**

- Baseline: no localization detection
- Improved: no localization detection
- Result: unresolved

**Test 05**

- Baseline confidence: approximately **0.0802**
- Improved confidence: approximately **0.6808**
- Baseline required a very low confidence threshold
- Improved model detected the plate at the normal 0.25 threshold
- Result: **major practical improvement**

**Test 06**

- Baseline localization was weak and incomplete
- Improved localization remained low-confidence and produced
  overlapping/incomplete candidates
- Result: still unresolved/weak

Comparison evidence is stored in:

```text
results/localization/baseline_vs_improved_report.txt
results/localization/comparison/
```

### Final Model Decision

The **improved YOLO11n localization model was selected as the final
localization model** because it provided:

- higher held-out recall
- higher held-out mAP@0.50
- higher held-out mAP@0.50:0.95
- a major generalization improvement on difficult Test 05
- only a very small precision reduction

The baseline model remains preserved as:

```text
models/localization/localization_baseline_best.pt
```

---

## 15. Final End-to-End Verification

The selected improved localization model and final character recognition
model were tested together through the complete recognition pipeline.

A final evaluation image produced:

```text
Status: success
Localization confidence: approximately 0.8722
Characters detected: 9

dhaka metro ga
147143
```

The annotated final detection image was also saved during the final
evaluation.

### Additional New Unseen Image

A separate newly uploaded vehicle image, not used in the preceding
training/evaluation workflow, was tested with the final models.

Result:

```text
Status: success
Localization confidence: 0.8879
Characters detected: 9

dhaka metro ga
144662
```

The localization bounding box visually covered the license plate
correctly.

The annotated image is stored in:

```text
inference_tests/final_new_image_test/test_vehicle_detected.jpg
```

The recognized text is model output; localization correctness was
additionally checked visually.

---

## 16. Local Windows/VS Code Verification

The final improved localization model was transferred from Google Drive
to the local project and tested with Python 3.11.

Three existing local test images completed end-to-end inference
successfully.

### Local Test 01

```text
Status: success
Localization confidence: 0.9213
Characters detected: 9

dhaka metro ca
160385
```

### Local Test 02

```text
Status: success
Localization confidence: 0.8885
Characters detected: 9

dhaka metro ga
335133
```

### Local Test 03

```text
Status: success
Localization confidence: 0.8837
Characters detected: 9

dhaka metro ga
254444
```

This confirms that the selected final models and pipeline operate
independently of the Google Colab runtime.

---

## 17. Project Structure

```text
Bangladesh_License_Plate_Recognition/
│
├── final_pipeline/
│   ├── lpr_pipeline.py
│   └── pipeline_info.txt
│
├── inference_tests/
│   ├── local_test/
│   └── final_new_image_test/
│       └── test_vehicle_detected.jpg
│
├── models/
│   ├── character/
│   │   └── character_best.pt
│   └── localization/
│       ├── localization_best.pt
│       └── localization_baseline_best.pt
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├──           02_localization_improvement.ipynb
│   ├── 03_colab_training_evaluation.ipynb
│   └── 04_final_local_inference.ipynb
│
├── results/
│   ├── character/
│   ├── final_evaluation/
│   ├── localization/
│   │   ├── localization_test/
│   │   ├── localization_improved_test/
│   │   ├── comparison/
│   │   └── baseline_vs_improved_report.txt
│   └── localization_dataset_diagnosis.txt
│
├── .gitignore
├── README.md
└── requirements.txt
```

Large datasets, virtual environments, and dataset ZIP archives are
intentionally excluded from Git version control.

---

## 18. Installation

Python 3.11 is recommended.

Create a virtual environment:

```bash
python -m venv .venv311
```

Activate it on Windows PowerShell:

```powershell
.\.venv311\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Core dependencies include:

- Ultralytics
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pillow

---

## 19. Using the Pipeline

Example:

```python
from final_pipeline.lpr_pipeline import BangladeshLPR

lpr = BangladeshLPR(
    "models/localization/localization_best.pt",
    "models/character/character_best.pt"
)

result = lpr.recognize("path/to/vehicle_image.jpg")

print(result["status"])
print(result.get("text"))
print(result.get("plate_box"))
print(result.get("localization_detections"))
print(result.get("characters"))
```

The returned dictionary can contain:

- `status`
- `text`
- `plate_box`
- `localization_detections`
- `characters`
- `crop`

The pipeline currently runs on CPU locally and can also be used in
Google Colab.


---

## 19.1 Final Local Inference Verification

A final interactive inference notebook was created for testing unseen vehicle images locally:

```text
notebooks/04_final_local_inference.ipynb

---
## Confusion Matrix Analysis

Confusion matrices were generated during held-out evaluation for both the character recognition and license plate localization models.

### Character Recognition Confusion Matrix

The character recognition evaluation includes both standard and normalized confusion matrices:

```text
results/character/character_test/confusion_matrix.png
results/character/character_test/confusion_matrix_normalized.png
```

The normalized confusion matrix shows strong recognition for several numeric, regional, and plate-prefix classes. Some less represented and visually similar Bangla character classes show greater cross-class confusion.

This is consistent with the final character-model performance:

| Metric | Score |
|---|---:|
| Precision | 0.933 |
| Recall | 0.918 |
| mAP50 | 0.944 |
| mAP50-95 | 0.708 |

During final unseen-image testing, one plate containing `ka` was initially recognized as `ha`. Further investigation showed that direct PIL input to the character detector produced the correct higher-confidence `ka` candidate for that image.

The final pipeline therefore uses:

```python
source=plate_crop
```

instead of:

```python
source=np.array(plate_crop)
```

After this change, the problematic plate was correctly recognized as:

```text
chattogram metro ka
110831
```

Regression testing on four previously successful images showed that their recognition results remained correct.

This example demonstrates why confusion-matrix analysis and unseen-image testing are both important: aggregate evaluation metrics can be strong while individual visually similar or sparsely represented character classes can still present challenges.

### Localization Confusion Matrices

Confusion matrices are also available for both localization experiments.

Baseline localization:

```text
results/localization/localization_test/confusion_matrix.png
results/localization/localization_test/confusion_matrix_normalized.png
```

Improved localization:

```text
results/localization/localization_improved_test/confusion_matrix.png
results/localization/localization_improved_test/confusion_matrix_normalized.png
```

These evaluation artifacts are retained in the repository so that the baseline and improved localization models can be compared alongside their quantitative test metrics.

---

## 20. Current Limitations

The improved model strengthens localization generalization but does not
eliminate all difficult cases.

Remaining challenges include:

- severe blur
- very low-resolution plates
- poor illumination
- unusual viewing angles
- partially visible plates
- large domain differences from the training dataset
- occasional incomplete or overlapping localization boxes
- sparse character classes with fewer examples

Difficult Tests 03 and 04 remained undetected after the targeted
localization improvement. Test 06 also remained weak.

Therefore, benchmark scores should not be interpreted as perfect
real-world recognition. Further improvement would require additional
diverse real-world Bangladesh vehicle images, especially examples
representing the unresolved conditions.

---

## 21. Experiment Reproducibility

The project retains the main evidence required to reproduce and inspect
the ML workflow:

```text
notebooks/01_dataset_audit.ipynb
notebooks/02_localization_improvement.ipynb
notebooks/03_colab_training_evaluation.ipynb

results/localization/localization_test/
results/localization/localization_improved_test/
results/localization/comparison/
results/localization/baseline_vs_improved_report.txt

models/localization/localization_baseline_best.pt
models/localization/localization_best.pt
models/character/character_best.pt
```

The original datasets are preserved separately and were not modified by
the final pipeline.

---

## 22. Next Steps

The core ML project is now complete.

Next project stages:

1.  Perform a final repository cleanup and Git status check.
2.  Publish the completed ML project to GitHub.
3.  After the ML repository is finalized, develop the web application
    separately around the verified inference pipeline.

Additional model/data improvements can be performed later as a separate
experiment rather than blocking completion of the current ML project.

---

## 23. Notes

The original datasets are preserved locally and are not modified by the
final pipeline.

Large datasets and dataset ZIP files are excluded from the repository to
keep the GitHub project lightweight.

The repository represents the completed ML stage with the **improved
localization model selected as the final model**, while the baseline
model and comparative evaluation evidence are retained for
reproducibility.
