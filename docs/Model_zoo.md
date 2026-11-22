# Model Zoo

[Back to main README](../README.md)

[Original models](#original-segmentation-models) · [V3 spine model](#v3-spine-model) · [Benchmark lesion models](#benchmark-lesion-models) · [Scientific context](#scientific-context) · [Citations](#citing-the-models-and-software)

## Original segmentation models

[Spinal-Multiple-Myeloma-SEG_nnUNet_models](https://doi.org/10.5281/zenodo.18598645), version 1.0.0, is the dataset-preparation collection. It contains conventional-CT spine models trained on VerSe20 or on combined VerSe20 and multiple myeloma data. Its lesion models use VMI 40 keV, with a binary spine mask as the second channel, and were produced during the iterative annotation project. The record includes a README describing the training cohorts.

| Dataset directory | Purpose | Used by the original scripts |
|---|---|---|
| `Dataset800_Spine_segmentation_trained_on_VerSe20` | Early ConvCT spine model | Archive only |
| `Dataset801_Spine_segmentation_trained_on_VerSe20` | Updated ConvCT spine baseline | Archive only |
| `Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together` | Combined-cohort ConvCT spine model | V1 and V2 |
| `Dataset650_MM_lesions_seg_nnUNet_v_2_0_VMI_40` | Initial iterative lesion model | Archive only |
| `Dataset651_MM_lesions_seg_nnUNet_v_2_1_VMI_40` | Intermediate iterative lesion model | Archive only |
| `Dataset652_MM_lesions_seg_nnUNet_v_2_2_VMI_40` | Later iterative lesion model | Archive only |
| `Dataset710_MM_Lesion_seg_just_VMI_40` | Original model trained on the complete annotated lesion dataset | V1 and V2 |

For V1/V2, manually extract `Spinal-Multiple-Myeloma-SEG_nnUNet_models.zip` and pass the parent of the two dataset directories as `--path_to_nnunet_results`. Each selected model uses this layout:

```text
original-models/<dataset>/
└── nnUNetTrainer__nnUNetPlans__3d_fullres/
    ├── dataset.json
    ├── plans.json
    └── fold_all/checkpoint_final.pth
```

Original 710 and benchmark 710 share a dataset name. The original scripts expect a full-dataset `fold_all` checkpoint, while V3 uses the separately published benchmark folds 0–4. Keep their model roots separate.

## V3 spine model

V3 uses **802**, `Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together`, from the [original Zenodo collection](https://doi.org/10.5281/zenodo.18598645). The single input is ordinary ConvCT and the nnU-Net channel key is `konvCT`. It uses `fold_all/checkpoint_final.pth` under `nnUNetTrainer__nnUNetPlans__3d_fullres`. Its output contains background (`0`) and anatomical spine labels (`1–28`). The installed model's `dataset.json` defines the label names and IDs. V3 preserves the anatomical labels and turns nonzero voxels into `1` for the binary channel consumed by the lesion model.

V3 can retrieve and verify the model automatically. With a supplied spine segmentation or prepared lesion channels it bypasses 802. See [V3 model storage](V3_benchmark_prediction.md#model-installation-and-cache).

## Benchmark lesion models

[Spinal-Multiple-Myeloma-SEG_nnUNet_models_benchmarking](https://doi.org/10.5281/zenodo.19708429), version 1.0.0, distributes one ZIP per dataset plus `splits_final.json`. The archive for each row is named `<dataset name>.zip`. Every model uses `nnUNetTrainer__nnUNetPlans__3d_fullres` with checkpoints `fold_0` through `fold_4`. The final input channel is always the binary spine mask, `mask_spine` (`0` background, `1` spine). Output labels are `0` background and `1` lesion.

| Model IDs | Training configuration | Total channels, including mask |
|---|---|---:|
| 700–707 | Leave-one-out retrained models | 8 |
| 708 | All eight CT reconstructions | 9 |
| 709–716 | Single-reconstruction models | 2 |
| 717 | All VMI reconstructions | 4 |
| 718 | All calcium-suppressed reconstructions | 5 |

A check mark identifies an input to the lesion model. ConvCT may also be needed separately when V3 runs automatic spine prediction.

| ID | nnU-Net dataset | ConvCT | VMI 40 | VMI 80 | VMI 120 | CaSupp 25 | CaSupp 50 | CaSupp 75 | CaSupp 100 | Channels |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| 700 | `Dataset700_MM_Lesion_seg_Leave_One_Out_without_ConvCT` | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 8 |
| 701 | `Dataset701_MM_Lesion_seg_Leave_One_Out_without_VMI_40` | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 8 |
| 702 | `Dataset702_MM_Lesion_seg_Leave_One_Out_without_VMI_80` | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | 8 |
| 703 | `Dataset703_MM_Lesion_seg_Leave_One_Out_without_VMI_120` | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | 8 |
| 704 | `Dataset704_MM_Lesion_seg_Leave_One_Out_without_CaSupp_25` | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | 8 |
| 705 | `Dataset705_MM_Lesion_seg_Leave_One_Out_without_CaSupp_50` | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | 8 |
| 706 | `Dataset706_MM_Lesion_seg_Leave_One_Out_without_CaSupp_75` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | 8 |
| 707 | `Dataset707_MM_Lesion_seg_Leave_One_Out_without_CaSupp_100` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | 8 |
| 708 | `Dataset708_MM_Lesion_seg_all_together` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 9 |
| 709 | `Dataset709_MM_Lesion_seg_just_ConvCT` | ✓ | — | — | — | — | — | — | — | 2 |
| 710 | `Dataset710_MM_Lesion_seg_just_VMI_40` | — | ✓ | — | — | — | — | — | — | 2 |
| 711 | `Dataset711_MM_Lesion_seg_just_VMI_80` | — | — | ✓ | — | — | — | — | — | 2 |
| 712 | `Dataset712_MM_Lesion_seg_just_VMI_120` | — | — | — | ✓ | — | — | — | — | 2 |
| 713 | `Dataset713_MM_Lesion_seg_just_CaSupp_25` | — | — | — | — | ✓ | — | — | — | 2 |
| 714 | `Dataset714_MM_Lesion_seg_just_CaSupp_50` | — | — | — | — | — | ✓ | — | — | 2 |
| 715 | `Dataset715_MM_Lesion_seg_just_CaSupp_75` | — | — | — | — | — | — | ✓ | — | 2 |
| 716 | `Dataset716_MM_Lesion_seg_just_CaSupp_100` | — | — | — | — | — | — | — | ✓ | 2 |
| 717 | `Dataset717_MM_Lesion_seg_all_VMI` | — | ✓ | ✓ | ✓ | — | — | — | — | 4 |
| 718 | `Dataset718_MM_Lesion_seg_all_CaSupp` | — | — | — | — | ✓ | ✓ | ✓ | ✓ | 5 |

### Exact channel order

Read each row from left to right starting at nnU-Net index `0000`. The dataset table and channel order agree with the [benchmark configuration tables](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking/tree/main/results/tables/model_configuration). Check the `dataset.json` bundled with the selected checkpoint when using nnU-Net directly.

| ID | Channels from index 0 |
|---:|---|
| 700 | `VMI_40`, `VMI_80`, `VMI_120`, `CaSupp_25`, `CaSupp_50`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |
| 701 | `ConvCT`, `VMI_80`, `VMI_120`, `CaSupp_25`, `CaSupp_50`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |
| 702 | `ConvCT`, `VMI_40`, `VMI_120`, `CaSupp_25`, `CaSupp_50`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |
| 703 | `ConvCT`, `VMI_40`, `VMI_80`, `CaSupp_25`, `CaSupp_50`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |
| 704 | `ConvCT`, `VMI_40`, `VMI_80`, `VMI_120`, `CaSupp_50`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |
| 705 | `ConvCT`, `VMI_40`, `VMI_80`, `VMI_120`, `CaSupp_25`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |
| 706 | `ConvCT`, `VMI_40`, `VMI_80`, `VMI_120`, `CaSupp_25`, `CaSupp_50`, `CaSupp_100`, `mask_spine` |
| 707 | `ConvCT`, `VMI_40`, `VMI_80`, `VMI_120`, `CaSupp_25`, `CaSupp_50`, `CaSupp_75`, `mask_spine` |
| 708 | `ConvCT`, `VMI_40`, `VMI_80`, `VMI_120`, `CaSupp_25`, `CaSupp_50`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |
| 709 | `ConvCT`, `mask_spine` |
| 710 | `VMI_40`, `mask_spine` |
| 711 | `VMI_80`, `mask_spine` |
| 712 | `VMI_120`, `mask_spine` |
| 713 | `CaSupp_25`, `mask_spine` |
| 714 | `CaSupp_50`, `mask_spine` |
| 715 | `CaSupp_75`, `mask_spine` |
| 716 | `CaSupp_100`, `mask_spine` |
| 717 | `VMI_40`, `VMI_80`, `VMI_120`, `mask_spine` |
| 718 | `CaSupp_25`, `CaSupp_50`, `CaSupp_75`, `CaSupp_100`, `mask_spine` |

## Scientific context

The [benchmarking repository](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking) documents training configuration, patient-grouped five-fold cross-validation, fold-wise evaluation and numerical analysis. The 19 configurations were trained with nnU-Net v2 in `3d_fullres` mode using the standard `nnUNetTrainer` for 1,000 epochs. The benchmark assessed each fold on its held-out examinations and used manually corrected vertebral masks. V3 runs all five trained lesion folds as an ensemble on a case, using automatic spine masks by default or masks supplied by the user. V3's application crop uses five voxels of padding. The public benchmark materials describe prepared data and nnU-Net preprocessing, but do not establish an exact corresponding training crop margin. Treat fold-wise scores separately from the practical V3 workflow.

The leave-one-out models were retrained with one reconstruction omitted. The benchmark also evaluated zero-input ablations using an existing trained model with one channel set to zero. The latter experiments do not define additional trained weights or model IDs. The [conceptual benchmark illustration](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking/blob/main/assets/benchmark_workflow.png) shows the experiment and evaluation workflow using schematic inputs and masks.

## Installation and usage

Use the manual original archive for V1/V2 and the [managed V3 cache](V3_benchmark_prediction.md#model-installation-and-cache) for benchmark inference. V3 resolves only the models required by the selected mode. Its `--offline` option uses verified installed weights. Inspect the catalog without downloading weights:

```bash
uv run --locked spinal-run-nnunet-benchmark-prediction --list_models
uv run --locked spinal-run-nnunet-benchmark-prediction --model_info 710
```

For standalone nnU-Net usage, extract the selected trainer folder and create aligned `<case>_0000.nii.gz`, `<case>_0001.nii.gz`, etc., in the verified channel order. With original checkpoints use `-f all`. With a benchmark lesion model use `-f 0 1 2 3 4`. The [standalone inference guide](https://github.com/MISAG-BUT/Spinal-Multiple-Myeloma-SEG-models-benchmarking#run-a-trained-model-on-new-data) shows how to invoke nnU-Net directly. It assumes the channels have already been prepared.

## Citing the models and software

Cite the [dataset paper and TCIA dataset](../README.md#citation), along with the record for the weights actually used. Cite the benchmark software record for its evaluation code or numerical results. The two weight records and the benchmarking software record list Apache 2.0. The TCIA dataset has its own terms.

| Resource | Citation source |
|---|---|
| Original model weights | [Zenodo 18598645](https://doi.org/10.5281/zenodo.18598645) |
| Benchmark lesion model weights | [Zenodo 19708429](https://doi.org/10.5281/zenodo.19708429) |
| Benchmark software and numerical results, v1.0.0 | [Zenodo 24035616](https://doi.org/10.5281/zenodo.24035616) |
| nnU-Net | [Isensee et al., *Nature Methods* 18, 203–211 (2021)](https://doi.org/10.1038/s41592-020-01008-z) |
| LongiSeg | [Software adapted for benchmark evaluation](https://github.com/MIC-DKFZ/LongiSeg), outside the prediction runtime |

<details>
<summary>BibTeX for the Zenodo records and nnU-Net</summary>

```bibtex
@misc{nohel_2026_18598645,
  author       = {Nohel, Michal and
                  Valek, Vlastimil and
                  Rohan, Tomas and
                  Stork, Martin and
                  Jakubicek, Roman and
                  Chmelik, Jiri and
                  Dostál, Marek},
  title        = {Spinal-Multiple-Myeloma-SEG\_nnUNet\_models},
  month        = feb,
  year         = 2026,
  publisher    = {Brno University of Technology},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.18598645},
  url          = {https://doi.org/10.5281/zenodo.18598645},
}

@misc{nohel_2026_19708429,
  author       = {Nohel, Michal and
                  Valek, Vlastimil and
                  Jakubicek, Roman and
                  Dostál, Marek and
                  Chmelik, Jiri},
  title        = {Spinal-Multiple-Myeloma-SEG\_nnUNet\_models\_benchmarking},
  month        = oct,
  year         = 2026,
  publisher    = {Brno University of Technology},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.19708429},
  url          = {https://doi.org/10.5281/zenodo.19708429},
}

@software{nohel_2026_24035616,
  author       = {Nohel, Michal and
                  Valek, Vlastimil and
                  Jakubicek, Roman and
                  Dostál, Marek and
                  Chmelik, Jiri},
  title        = {Spinal-Multiple-Myeloma-SEG models benchmarking},
  month        = nov,
  year         = 2026,
  publisher    = {Brno University of Technology},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24035616},
  url          = {https://doi.org/10.5281/zenodo.24035616},
}

@article{isensee2021nnunet,
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  title   = {{nnU-Net}: a self-configuring method for deep learning-based biomedical image segmentation},
  journal = {Nature Methods},
  volume  = {18},
  pages   = {203--211},
  year    = {2021},
  doi     = {10.1038/s41592-020-01008-z}
}
```

</details>

[Installation](Installation.md) · [V1 TCIA prediction](V1_TCIA_prediction.md) · [V2 NIfTI prediction](V2_new_data_prediction.md) · [V3 benchmark prediction](V3_benchmark_prediction.md)
