# Effect of an Intraoral Image Preprocessing Pipeline on Prompt-Conditioned Dental Caries Segmentation with SAM 2

Code, executed notebooks and results for the paper **"Efecto de un pipeline de preprocesamiento de imágenes intraorales sobre la segmentación de caries dentales condicionada por prompt mediante SAM 2"** (English: *Effect of an intraoral image preprocessing pipeline on prompt-conditioned dental caries segmentation with SAM 2*).

Universidad Técnica de Machala, Faculty of Civil Engineering, Information Technology Engineering Program (Machala, Ecuador).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23458622.svg)](https://doi.org/10.5281/zenodo.23458622) *(fine-tuned checkpoints)*

> **Disclaimer.** This is research material. Models were evaluated with a bounding-box prompt derived from the ground-truth annotation (*oracle prompt*) and are **not validated for clinical or diagnostic use**.

> **Language note.** The paper and the notebooks (code comments and outputs) are in Spanish. This README is in English.

---

## 1. Summary

Public dental datasets mix intraoral photographs taken with different devices and protocols (illumination, noise, blur, specular highlights). This work tests whether a **deterministic preprocessing pipeline**, applied before fine-tuning, changes how **SAM 2** segments caries.

The contribution is not a new architecture but the **pre-training stage** and a **paired experimental design**: two versions of the same corpus (Control and Experimental), SAM 2 partially fine-tuned under identical settings, and a per-lesion comparison replicated with three seeds (56, 42, 25).

**Main result.** The mean pipeline effect on Dice (primary metric), estimated with a mixed-effects model, was **−0.0031** (95% CI [−0.0097, 0.0035]; p = 0.358). No metric (of eleven) had a confidence interval excluding zero. The results are compatible with a small or null effect; they support neither equivalence nor practical relevance.

---

## 2. Experimental design

- **Framework:** KDD (Fayyad et al., 1996) in five phases: Selection, Preprocessing (structural audit), Transformation, Data Mining, Evaluation.
- **Paired design:** each lesion is its own control. Both conditions share the same split, ground-truth masks and training configuration; **the only varying factor is the pipeline**.
- **Prompt:** bounding box from the ground-truth contour (oracle prompt), so the comparison targets mask quality, not lesion localization.
- **Experimental unit:** the individual lesion (70.5% of images contain more than one lesion). Inference is clustered by image (cluster bootstrap, sign-flip permutation, mixed model with a random intercept per image).
- **Replicates:** seeds 56, 42, 25 on a single fixed split.

| Condition | Model input |
|---|---|
| **Control** | Original image, only letterboxed to 1024×1024 (required by SAM 2). |
| **Experimental** | Same image after the six-step pipeline, then the same letterbox. |

---

## 3. Data

| Source | Original format | License | Declared size |
|---|---|---|---|
| **DentalAI** (Valluri, 2023), Kaggle | Supervisely, multi-class | CC BY 4.0 | 2,495 images |
| **Dental Caries**, Roboflow Universe (v1) | YOLO segmentation (`Tooth`, `Caries`) | CC BY 4.0 | 1,991 images |

Downloaded on 2026-05-18. Links: [TODO: DentalAI Kaggle URL] · [TODO: Dental Caries Roboflow URL].

Both were converted to single-class YOLO segmentation (`Caries`); the `Cavity` and `Tooth` classes were dropped. Quality filtering (resolution, blur, exposure, duplicates by pHash, invalid polygons) plus a stricter geometry audit gave the final corpus:

| Split | Images | Lesions |
|---|---:|---:|
| Train | 727 | 2,455 |
| Validation | 156 | 587 |
| Test | 156 | 564 |
| **Total** | **1,039** | **3,606** |

(Stratified 70/15/15 split **by image**; full per-source counts are in the paper, Tables 1–2.)

---

## 4. Preprocessing pipeline (Experimental group)

Applied at original resolution, before the letterbox:

| # | Operation | Parameters |
|--:|---|---|
| 1 | White balance | Shades of Gray, p = 4.0 |
| 2 | CLAHE | L channel (LAB), clip = 1.3, 8×8 tiles |
| 3 | Specular highlights | Selective L smoothing; V ≥ 245, S ≤ 60, dilation 8, attenuation 0.7 |
| 4 | Denoising | Median, kernel 3 |
| 5 | Edge enhancement | Laplacian fusion, k = 7, weight 0.05 |
| 6 | Brightness/contrast | `convertScaleAbs`, brightness −5, α = 1.0 |

Letterbox to 1024×1024 uses `INTER_AREA` (downscale) / `INTER_CUBIC` (upscale), so geometric labels are identical in both groups.

---

## 5. Model and fine-tuning

- **Model:** SAM 2.1 Hiera Large (`sam2.1_hiera_large.pt`), commit `2b90b9f` of `facebookresearch/sam2`.
- **Strategy:** partial fine-tuning. Only the mask decoder is trained (11,737,142 parameters, 5.23% of 224,446,642).
- **Setup:** Dice 0.7 + Focal 0.3 loss; AdamW (lr 1e-4, wd 1e-4); cosine schedule; batch 4; **80 fixed epochs**, no early stopping; decoder EMA 0.999; bf16.
- **Evaluated model:** EMA of the **last epoch** (rule fixed before seeing results).
- **Prompt:** box with 4 px margin; ±5% jitter in training only.

Each notebook logs a configuration fingerprint so you can verify that Control and Experimental share exactly the same settings.

---

## 6. Evaluation

Six models (2 groups × 3 seeds) on the same **564 test lesions** (1,692 paired observations; 1,675 valid for HD95/ASSD). Eleven per-lesion metrics, Dice primary:

- **Overlap:** Dice, IoU, precision, recall, AP
- **Boundary:** Boundary IoU (2%), Boundary IoU (3 px), HD95, ASSD
- **Calibration (prompt ROI):** Brier, ECE

Statistics: paired differences with image-level cluster bootstrap (10,000) and sign-flip permutation (20,000); linear mixed model with random image intercept and sum-coded seed/source effects; Wald tests for seed heterogeneity and seed × source interaction; JZS Bayes factor; TOST with an empirical margin; Benjamini–Hochberg (BH) on the ten secondary metrics.

### Group means (3 seeds, n = 564 lesions per group and seed)

| Metric | Control | Experimental | Δ (Exp − Ctrl) |
|---|---:|---:|---:|
| Dice (primary) | 0.8011 ± 0.1391 | 0.7968 ± 0.1387 | −0.0042 |
| IoU | 0.6865 | 0.6804 | −0.0061 |
| Precision | 0.8307 | 0.8226 | −0.0081 |
| Recall | 0.8035 | 0.8005 | −0.0030 |
| AP | 0.7383 | 0.7321 | −0.0062 |
| Boundary IoU (2%) | 0.6787 | 0.6725 | −0.0062 |
| Boundary IoU (3 px) | 0.4357 | 0.4287 | −0.0069 |
| HD95 (px) | 4.6094 | 4.5103 | −0.0991 |
| ASSD (px) | 1.5117 | 1.5142 | +0.0025 |
| Brier (ROI) | 0.0698 | 0.0710 | +0.0012 |
| ECE (ROI) | 0.0693 | 0.0706 | +0.0012 |

### Key results

| Finding | Value |
|---|---|
| Mean effect on Dice (mixed model) | **−0.0031**, 95% CI [−0.0097, 0.0035], p = 0.358 |
| Metrics with CI excluding zero | None of 11. Lowest p: Boundary IoU (2%), p = 0.080 (BH-adjusted 0.443) |
| Sign consistency across seeds | 10 of 11 metrics (unfavorable to Experimental in 9; HD95 favorable). Only ASSD changes sign |
| Dice by seed (56 / 42 / 25) | −0.0039 / −0.0039 / −0.0049 |
| Seed heterogeneity (Dice) | χ²(2) = 0.02, p = 0.992 |
| Bayes factor (Dice, per seed) | BF₀₁ ≈ 6.5 / 6.4 / 5.2 (moderate support for no difference; not an equivalence test) |
| Precision by seed | −0.0091 / −0.0069 / −0.0082. CI excludes zero only for seed 25 ([−0.0163, −0.0004]); no contrast survives BH (adjusted p = 0.211). Pooled: −0.0059 (p = 0.177) |
| Effect-to-noise ratio (Control seed noise) | Dice 2.3; precision 3.8; ASSD 5.5 |
| Cross-seed agreement of the per-image effect (Dice) | Spearman 0.44 (mean over seed pairs); ICC(3,1) = 0.40 |
| By source (Dice) | DentalAI −0.0047 to −0.0075; Roboflow −0.0008 to −0.0027; always negative, never significant. Source difference −0.0059 (CI [−0.0286, 0.0169], p = 0.612); seed × source interaction χ²(2) = 0.27, p = 0.873 |
| Validation Dice (last-epoch EMA) | Control 0.8046 / 0.8077 / 0.8075; Experimental 0.7982 / 0.7971 / 0.7955; mean gap ≈ 0.010 |
| Excluded observations | 17 of 1,692 for HD95/ASSD (556 / 560 / 559 valid pairs per seed). Empty predictions: 13 (Experimental) vs 8 (Control) |
| Calibration | Lesion confidence deficit ≈ 0.20 in both groups; Brier/ECE ≈ 0.07 in the ROI; no relevant change |

**Conclusion.** A small or null effect, with neither significance nor equivalence established. The pattern (lower precision, stable recall) is compatible with slightly wider masks, which would need to be verified with predicted-area analysis.

### Exploratory pooled screening (DentalAI, seeds 7, 13, 91)

Four conditions (Control, full pipeline, photometric subset: white balance + CLAHE + brightness/contrast, spatial subset: specular highlights + median + edges) × 3 new seeds = 12 trainings on the 593 DentalAI images (89 test images / 342 test lesions). Same prompt, loss, optimizer and augmentations as the main experiment, but **up to 60 epochs with early stopping (patience 10 on validation Dice), best-validation checkpoint, no EMA**, so magnitudes are not comparable with the main experiment. It reproduced the same direction in Dice and hints that the precision drop may be associated with the photometric component. It is exploratory and does **not** allow attributing effects to individual operations.

Files: `Cribado/` (see Section 7).

---

## 7. Repository structure

Mirrors the folder layout the notebooks expect, so the repo can be copied to Google Drive and run without changing paths. Folders marked *(empty)* only preserve the structure.

```
KDD_Caries/
├── README.md
├── LICENSE
├── CITATION.cff
├── datasets_raw/                 (empty; see Section 8)
├── Fase 1/   KDD-Fase 1.ipynb, phase1_state.json, figuras/
├── Fase 2/   KDD-Fase 2.ipynb, final_corpus_summary.json,
│             Corpus/ (data.yaml, corpus_metadata.csv)
├── Fase 3/   KDD-Fase 3.ipynb, phase3_summary.json,
│             Salida/Control|Experimental/data.yaml
├── Fase 4/   Pesos-Base/ (empty),
│             KDD-Fase 4 Control - Seed 56|42|25.ipynb,
│             KDD-Fase 4 Experimental - Seed 56|42|25.ipynb,
│             Semilla N/Salida/<Group>/ (Logs, Figuras, Checkpoints (empty), metadata JSON)
├── Fase 5/   KDD-Fase 5.ipynb, Semillas/ (Tablas, Figuras)
└── Cribado/  Exploratory screening: registro_experimentos.csv,
              metricas_por_lesion_consolidado.csv, Historiales/ (12 JSON)
```

| Phase | Notebook | Reads | Produces |
|---|---|---|---|
| 1 Selection | `KDD-Fase 1.ipynb` | `datasets_raw/` | `phase1_state.json` |
| 2 Audit | `KDD-Fase 2.ipynb` | Phase 1 state | `Fase 2/Corpus/` (1,039 images, 3,606 lesions) |
| 3 Transformation | `KDD-Fase 3.ipynb` | `Fase 2/Corpus/` | Control and Experimental datasets |
| 4 Mining | `KDD-Fase 4 <Group> - Seed <N>.ipynb` (×6) | Phase 3 output, base weights | One checkpoint per model |
| 5 Evaluation | `KDD-Fase 5.ipynb` | 6 checkpoints, Phase 3 output, `corpus_metadata.csv` | Tables and figures |

Notebooks are published **with their outputs**, so they can be reviewed without re-running.

---

## 8. What is NOT in the repository

| Item | Why | How to get it |
|---|---|---|
| Original datasets (`datasets_raw/`) | Third-party, several GB | Download from the sources in Section 3 and place them as `Dentalai-Dataset.tar` and `Dental-Caries-Roboflow.zip` |
| Audited corpus images (`Fase 2/Corpus/train, valid, test`) | Full-resolution images (up to 8,688 px wide) | Regenerate with notebooks 1–2 |
| Preprocessed datasets (`Fase 3/Salida/*`) | Two 1024×1024 copies; deterministic | Regenerate with notebook 3 |
| SAM 2.1 base weights | Third-party | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` (notebooks 4–5 download them if missing) |
| **Fine-tuned checkpoints** (6 × ~900 MB) | Exceed GitHub limits | **Zenodo: [10.5281/zenodo.23458622](https://doi.org/10.5281/zenodo.23458622)**. Place in `Fase 4/Semilla N/Salida/<Group>/Checkpoints/` keeping the name `sam2_<run_id>_final_ema.pt`; verify with the `SHA256SUMS.txt` included in the record |
| Best-epoch and portable weights | Redundant / unused | Not published |
| Screening checkpoints, materialized images and notebook | Size; the notebook is a single job-queue notebook that ran all 12 trainings and its outputs are very long | Only the result tables and training histories are published (`Cribado/`) |

**About published checkpoints.** They were re-saved without the absolute paths stored in their internal config. Tensors are identical to the originals, but the file (and its SHA-256) differs. Each `metadatos.json` keeps the original hash (`sha256_checkpoint_final`) and the published one (`sha256_checkpoint_final_publicado`).

---

## 9. How to reproduce

**Requirements:** Google Colab (or equivalent GPU environment) and Google Drive. Phase 4 takes ~7 h per model (~42 h total). Software versions are in the paper (Section 3.2.2) and in the `versiones` field of each `metadatos.json`.

1. Copy the repo to `/content/drive/MyDrive/KDD_Caries/`.
2. Download the original datasets into `datasets_raw/`.
3. Run notebooks **in order** (1 → 2 → 3 → 4 → 5).
4. **To skip Phase 4:** download the 6 checkpoints from Zenodo into their `Checkpoints/` folders and run `KDD-Fase 5.ipynb` directly (it needs the output of notebooks 1–3).

Paths are centralized in the first configuration cell of each notebook (`BASE_DRIVE` or `OUTPUT_ROOT`).

### Phase 4 runs

| Seed | Group | `run_id` | Config fingerprint |
|---:|---|---|---|
| 56 | Control | `control_s56_20261001_042024` | `31eadb2a84e6` |
| 56 | Experimental | `experimental_s56_20261001_181520` | `31eadb2a84e6` |
| 42 | Control | `control_s42_20261002_033006` | `62e744b83283` |
| 42 | Experimental | `experimental_s42_20261002_214552` | `62e744b83283` |
| 25 | Control | `control_s25_20261003_055906` | `75fd408dae0f` |
| 25 | Experimental | `experimental_s25_20261015_191732` | `75fd408dae0f` |

The fingerprint differs across seeds (the seed is part of the configuration) but is identical between groups within a seed.

---

## 10. Limitations and transparency

- **Oracle prompt:** results do not reflect an end-to-end detection system or the pipeline's effect on lesion localization.
- **Pipeline evaluated as a unit:** no ablation, so effects cannot be attributed to any single operation.
- **No significance ≠ equivalence:** the TOST margin is empirical (between-seed noise of Control), not clinical.
- **Single split, three seeds:** cluster inference does not capture between-split variability.
- **Annotation quality:** ground truth was visually checked on only 9.6% of the corpus, by one author without specialized clinical training and without double review.
- **Two public sources** with uncontrolled capture protocols; results do not generalize to standardized-capture corpora.
- **Distance metrics:** HD95 and ASSD exclude empty predictions (more frequent in the Experimental group), so the direction of their effect cannot be established with this design.
- **Notebooks:** outputs are preserved; Drive paths were normalized to `KDD_Caries/...` and Colab account metadata was removed (no figures or images were altered). Original Phase 4–5 runs used a different folder layout. Phase 4 was not re-run for publication (~7 h each), so a re-run may not reproduce checkpoints bit by bit because the Colab environment changed between runs. Phases 1–3 were re-run when preparing this repo [TODO: confirm]; each notebook compares its counts with the paper and prints a warning if they differ.

---

## 11. License and attributions

- **Code and notebooks:** [MIT License](LICENSE) © 2026 Jonathan Medina, Roger Ramírez, Wilmer Rivas, Jennifer Célleri.
- **Source datasets:** DentalAI (Valluri, 2023, Kaggle) and Dental Caries (Roboflow Universe, v1), both **CC BY 4.0**. Derived files included here (`corpus_metadata.csv`, `data.yaml`, summaries, example figures) retain that attribution.
- **SAM 2:** Ravi et al. (2024), arXiv:2408.00714, released under its own license (see the official repository). The fine-tuned checkpoints derive from it and must comply with it.
- **Responsible use:** research artifacts only; not a substitute for professional judgment and not for clinical decisions.

---

## 12. Citation

If you use this work, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{medina2026sam2preprocessing,
  title   = {Efecto de un pipeline de preprocesamiento de im\'agenes intraorales sobre la segmentaci\'on de caries dentales condicionada por prompt mediante {SAM} 2},
  author  = {Medina, Jonathan and Ram\'irez, Roger and Rivas, Wilmer and C\'elleri, Jennifer},
  year    = {2026},
  note    = {Manuscript. Code: [TODO: repository URL]. Update with journal/conference and DOI once published.}
}
```

Fine-tuned checkpoints:

```bibtex
@dataset{medina2026sam2checkpoints,
  title     = {Fine-tuned SAM 2.1 checkpoints for caries segmentation (Control/Experimental, seeds 56, 42, 25)},
  author    = {Medina, Jonathan and Ram\'irez, Roger and Rivas, Wilmer and C\'elleri, Jennifer},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23458622}
}
```

Also cite SAM 2 (Ravi et al., 2024, arXiv:2408.00714) and the source datasets.

---

## 13. Authors and contact

Jonathan Medina (corresponding author), Roger Ramírez, Wilmer Rivas, Jennifer Célleri.
Universidad Técnica de Machala, Ecuador.

- Corresponding author: jmedina18@utmachala.edu.ec
- Research group (GIDCOWEB): jcelleri@utmachala.edu.ec
- Issues and questions about the code: please open a GitHub issue.
