# Spinal-Multiple-Myeloma-SEG

![Graphical abstract showing the original dataset creation pipeline and data structure](graphical_abstract.png)

**Authors:** Michal Nohel<sup>1,†</sup>, Vlastimil Valek<sup>2,3,†</sup>, Tomas Rohan<sup>2,3</sup>, Martin Stork<sup>4</sup>, Roman Jakubicek<sup>1</sup>, Jiri Chmelik<sup>1,††,\*</sup> , Marek Dostal<sup>2,5,††,\*</sup>

**Author Affiliations:**<br>
<sup>1</sup> Department of Biomedical Engineering, Faculty of Electrical Engineering and Communication, Brno University of Technology, Brno, Czech Republic<br>
<sup>2</sup> Department of Radiology and Nuclear Medicine, University Hospital Brno, Brno, Czech Republic<br>
<sup>3</sup> Department of Radiology and Nuclear Medicine, Faculty of Medicine, Masaryk University, Brno, Czech Republic<br>
<sup>4</sup> Internal Hematology and Oncology Clinic, University Hospital Brno, Brno, Czech Republic<br>
<sup>5</sup> Department of Biophysics, Masaryk University, Brno, Czech Republic<br>

<sup>*</sup> Corresponding authors: Marek Dostal (Dostal.Marek@fnbrno.cz), Jiri Chmelik (chmelikj@vut.cz)<br>
<sup>†</sup> These authors contributed equally to this work<br>
<sup>††</sup> These authors contributed equally to this work

## Overview

This repository accompanies the published *Scientific Data* article [**Spinal-Multiple-Myeloma-SEG: Segmentation of spinal multiple myeloma lesions in dual-energy CT**](https://doi.org/10.1038/s41597-026-08061-x). It provides [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)-based tools for automatic spine and osteolytic multiple myeloma lesion segmentation, plus an interactive dataset viewer.

The [TCIA dataset](https://doi.org/10.7937/k4qv-hh78) contains 72 dual-energy CT examinations from 67 patients, with conventional CT, virtual monoenergetic and calcium-suppressed reconstructions and segmentation annotations. The original workflows use the models developed for dataset preparation. The V3 workflow brings the trained spectral benchmark models to practical inference on aligned NIfTI data.

## Publications and research resources

| Resource | What it provides |
|---|---|
| [Scientific Data article, Nohel et al. (2026)](https://doi.org/10.1038/s41597-026-08061-x) | Dataset description and annotation workflow |
| [TCIA dataset, version 1](https://doi.org/10.7937/k4qv-hh78) | DECT images and annotations |
| [Original model weights](https://doi.org/10.5281/zenodo.18598645) | Spine, vertebra and lesion models from dataset preparation |
| [Original repository snapshot](https://zenodo.org/records/19369118) · [Software DOI](https://doi.org/10.5281/zenodo.15878952) | Archived inference and visualization software |
| [Dataset extension](https://doi.org/10.59275/j.melba.2026-d741) · [Extended dataset](https://doi.org/10.5281/zenodo.21628232) | Expert-validated lumbar trabecular bone masks |
| [Benchmarking repository](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking) · [Software and results, v1.0.0](https://doi.org/10.5281/zenodo.24035616) | Cross-validation, evaluation, statistics, and reproduction of benchmark tables and figures |
| [Benchmark model weights](https://doi.org/10.5281/zenodo.19708429) | 19 trained lesion configurations, five folds each and the patient-grouped split |

### Spectral reconstruction benchmark

*Benchmarking Spectral Reconstructions (VMI and Calcium-Suppressed) for Multiple Myeloma Segmentation on Dual-Energy CT* is a manuscript in preparation for submission to *Medical Image Analysis*, according to the [benchmarking project](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking). It compares DECT reconstruction choices for lesion segmentation. The companion repository documents the experiments and numerical results. This repository provides the prediction workflows and viewer.

**Benchmark authors:** Michal Nohel<sup>1,⁎</sup>, Vlastimil Valek<sup>2,3</sup>, Roman Jakubicek<sup>1</sup>, Marek Dostal<sup>2,4</sup>, Jiri Chmelik<sup>1</sup>.

<sup>1</sup> Department of Biomedical Engineering, Faculty of Electrical Engineering and Communication, Brno University of Technology, Brno, Czech Republic<br>
<sup>2</sup> Department of Radiology and Nuclear Medicine, University Hospital Brno, Brno, Czech Republic<br>
<sup>3</sup> Department of Radiology and Nuclear Medicine, Faculty of Medicine, Masaryk University, Brno, Czech Republic<br>
<sup>4</sup> Department of Biophysics, Masaryk University, Brno, Czech Republic<br>
<sup>⁎</sup> Corresponding author: Michal Nohel. [Benchmark author information](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking#readme)

## Available workflows

| Workflow | Input | Models | Use |
|---|---|---|---|
| [V1 · TCIA dataset segmentation](docs/V1_TCIA_prediction.md) | DICOM | Original | Convert TCIA series and segment spine and lesions |
| [V2 · New data segmentation](docs/V2_new_data_prediction.md) | Aligned RAS NIfTI | Original | Segment custom ConvCT and VMI 40 keV pairs |
| [V3 · Benchmark model segmentation](docs/V3_benchmark_prediction.md) | Aligned NIfTI | Spine 802 and benchmark lesion 700–718 | Select spectral reconstructions and run spine, lesion or combined prediction |
| [Dataset Viewer](docs/Dataset_viewer.md) | DICOM and NIfTI masks | None | Explore TCIA reconstructions and segmentation overlays in napari |

V1, V2 and V3 are workflow names. See the [Model Zoo](docs/Model_zoo.md) for the manually installed original checkpoints, V3 automatic downloads, and the exact channel order for each benchmark model.

## Installation and quick start

The locked installation targets Linux x86-64 with Python 3.12 and an NVIDIA CUDA 13.0-compatible driver. From a checkout containing these guides:

```bash
uv sync --locked
uv run --locked spinal-run-nnunet-benchmark-prediction --help
```

uv manages the project's `.venv`. Add the GUI extra for the viewer:

```bash
uv sync --locked --extra gui
uv run --locked --extra gui spinal-db-viewer --help
```

Keep `--extra gui` on subsequent synchronizing uv commands if you use the viewer. See [Installation](docs/Installation.md) for cloning, pip installation and VS Code setup.

| Python script in the repository root | Installed launcher |
|---|---|
| [`run_prediction_of_nnUNet_networks_on_TCIA_data_final.py`](run_prediction_of_nnUNet_networks_on_TCIA_data_final.py) | `spinal-run-nnunet-TCIA` |
| [`run_prediction_of_nnUNet_networks_on_new_data.py`](run_prediction_of_nnUNet_networks_on_new_data.py) | `spinal-run-nnunet-new-prediction` |
| [`run_prediction_of_nnUNet_benchmark_models.py`](run_prediction_of_nnUNet_benchmark_models.py) | `spinal-run-nnunet-benchmark-prediction` |
| [`Database_viewer_final.py`](Database_viewer_final.py) | `spinal-db-viewer` |

Run the script with `.venv/bin/python`, use its launcher or open it in VS Code with the project interpreter. [`config.py`](config.py) supplies V1, V2 and viewer defaults. For V3, pass case inputs, prediction mode and output destination as arguments.

## Citation

If you use this repository or the associated dataset, please cite the published *Scientific Data* article and the original TCIA dataset.

**Scientific Data article:**

Nohel, M., Valek, V., Rohan, T., Stork, M., Jakubicek, R., Chmelik, J., & Dostal, M. (2026). **Spinal-Multiple-Myeloma-SEG: Segmentation of spinal multiple myeloma lesions in dual-energy CT.** *Scientific Data*. https://doi.org/10.1038/s41597-026-08061-x

```bibtex
@article{nohel2026spinalmm,
  author  = {Nohel, Michal and Valek, Vlastimil and Rohan, Tomas and Stork, Martin and Jakubicek, Roman and Chmelik, Jiri and Dostal, Marek},
  title   = {{Spinal-Multiple-Myeloma-SEG}: Segmentation of spinal multiple myeloma lesions in dual-energy {CT}},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-08061-x},
  url     = {https://doi.org/10.1038/s41597-026-08061-x}
}
```

**Original TCIA dataset:**

Nohel, M., Válek V., Rohan T., Štork M., Jakubíček R., Chmelík J., Dostál M. (2026) **Segmentation of spinal multiple myeloma lesions in dual-energy CT (Spinal-Multiple-Myeloma-SEG) (Version 1) [Dataset]**. The Cancer Imaging Archive. https://doi.org/10.7937/k4qv-hh78

```bibtex
@misc{Nohel_Valek_Rohan_Stork_Jakubicek_Chmelik_Dostál_2026,
author={Nohel, Michal and Valek, Vlastimil and Rohan, Tomas and Stork, Martin and Jakubicek, Roman and Chmelik, Jiri and Dostál, Marek},
title={Segmentation of spinal multiple myeloma lesions in dual-energy CT (Spinal-Multiple-Myeloma-SEG)},
url={https://www.cancerimagingarchive.net/collection/Spinal-Multiple-Myeloma-SEG},
DOI={10.7937/K4QV-HH78},
publisher={The Cancer Imaging Archive},
year={2026},
version={1}
}
```

For model-specific and software citations, see the [Model Zoo](docs/Model_zoo.md#citing-the-models-and-software). See the [TCIA collection](https://doi.org/10.7937/k4qv-hh78) for related publications and resources.

## Acknowledgements

The dataset paper acknowledges a workstation loan from Philips Healthcare, computing resources from e-INFRA CZ (ID 90254) supported by the Czech Ministry of Education, Youth and Sports, and Czech Ministry of Health institutional support for University Hospital Brno (FNBr, 65269705). [Published acknowledgements](https://doi.org/10.1038/s41597-026-08061-x)

## License and research use

The source code is distributed under the [Apache License 2.0](LICENSE). Consult the [TCIA collection](https://doi.org/10.7937/k4qv-hh78) for its CC BY 4.0 data terms and the individual Zenodo records for model terms. The workflows and models are provided for research use. Technical verification of predictions does not establish clinical validity.
