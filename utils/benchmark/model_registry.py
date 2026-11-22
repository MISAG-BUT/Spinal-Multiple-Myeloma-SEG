"""Pinned V3 model metadata verified from Zenodo ZIP directories and JSONs.

Sources: https://zenodo.org/api/records/19708429 and
https://zenodo.org/api/records/18598645. No network access occurs on import.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class ModelSpec:
    dataset_id: int
    dataset_name: str
    zip_filename: str
    zenodo_record_id: int
    zip_size: int
    md5: str
    channel_names: Tuple[str, ...]
    required_modalities: Tuple[str, ...]
    folds: Tuple[Union[int, str], ...]
    trainer_directory: str = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    checkpoint_filename: str = "checkpoint_final.pth"
    expected_nnunet_version: Optional[str] = "2.6.4"
    binary_mask_channel: Optional[str] = "mask_spine"
    archive_root: str = ""
    release_version: str = "1.0.0"

    @property
    def download_url(self):
        return "https://zenodo.org/api/records/{}/files/{}/content".format(
            self.zenodo_record_id, self.zip_filename
        )

    @property
    def archive_model_subtree(self):
        return self.archive_root + self.dataset_name + "/"


# Channel order is explicitly recorded for each model, never inferred from names.
_ROWS = (
    (700, "MM_Lesion_seg_Leave_One_Out_without_ConvCT", 1663265910, "2aad6226850e56312ff61a75c47868ac", ("VMI_40", "VMI_80", "VMI_120", "CaSupp_25", "CaSupp_50", "CaSupp_75", "CaSupp_100", "mask_spine")),
    (701, "MM_Lesion_seg_Leave_One_Out_without_VMI_40", 1663561505, "6ec6266e17017409053e23e4c4a13b57", ("ConvCT", "VMI_80", "VMI_120", "CaSupp_25", "CaSupp_50", "CaSupp_75", "CaSupp_100", "mask_spine")),
    (702, "MM_Lesion_seg_Leave_One_Out_without_VMI_80", 1663161787, "d74d55511b03a56239c136c85db503ed", ("ConvCT", "VMI_40", "VMI_120", "CaSupp_25", "CaSupp_50", "CaSupp_75", "CaSupp_100", "mask_spine")),
    (703, "MM_Lesion_seg_Leave_One_Out_without_VMI_120", 1663181666, "e05984dd42e2cde771be04498861f585", ("ConvCT", "VMI_40", "VMI_80", "CaSupp_25", "CaSupp_50", "CaSupp_75", "CaSupp_100", "mask_spine")),
    (704, "MM_Lesion_seg_Leave_One_Out_without_CaSupp_25", 1663110777, "c1414d12820721b629fa25e7bebe4a0b", ("ConvCT", "VMI_40", "VMI_80", "VMI_120", "CaSupp_50", "CaSupp_75", "CaSupp_100", "mask_spine")),
    (705, "MM_Lesion_seg_Leave_One_Out_without_CaSupp_50", 1663068766, "9506101476df080b2728d8f6b8c31336", ("ConvCT", "VMI_40", "VMI_80", "VMI_120", "CaSupp_25", "CaSupp_75", "CaSupp_100", "mask_spine")),
    (706, "MM_Lesion_seg_Leave_One_Out_without_CaSupp_75", 1663168915, "24ee9bd5f20343a1d0eca497fa529c6a", ("ConvCT", "VMI_40", "VMI_80", "VMI_120", "CaSupp_25", "CaSupp_50", "CaSupp_100", "mask_spine")),
    (707, "MM_Lesion_seg_Leave_One_Out_without_CaSupp_100", 1663313071, "f6db04fd6afaf8923487220873b9ff03", ("ConvCT", "VMI_40", "VMI_80", "VMI_120", "CaSupp_25", "CaSupp_50", "CaSupp_75", "mask_spine")),
    (708, "MM_Lesion_seg_all_together", 1663289681, "b1e85426b75e250d346d8ca933c32849", ("ConvCT", "VMI_40", "VMI_80", "VMI_120", "CaSupp_25", "CaSupp_50", "CaSupp_75", "CaSupp_100", "mask_spine")),
    (709, "MM_Lesion_seg_just_ConvCT", 1662826000, "b135b0568d639dc346a371ccbca39e40", ("ConvCT", "mask_spine")),
    (710, "MM_Lesion_seg_just_VMI_40", 1662450334, "6bb8d72f7fe23564ce933d5771dfc1ea", ("VMI_40", "mask_spine")),
    (711, "MM_Lesion_seg_just_VMI_80", 1663281292, "978660c61a965a54e283155ae133bc14", ("VMI_80", "mask_spine")),
    (712, "MM_Lesion_seg_just_VMI_120", 1662850531, "2e918838e9b6c6766fb00bbd19c6bd22", ("VMI_120", "mask_spine")),
    (713, "MM_Lesion_seg_just_CaSupp_25", 1662354710, "aeaad14f33635504febf3fe8a3762022", ("CaSupp_25", "mask_spine")),
    (714, "MM_Lesion_seg_just_CaSupp_50", 1661928727, "33d2f4a782c09de02492382c74b691cb", ("CaSupp_50", "mask_spine")),
    (715, "MM_Lesion_seg_just_CaSupp_75", 1662056573, "116e7cf69f9b4b010183e792eb50fec1", ("CaSupp_75", "mask_spine")),
    (716, "MM_Lesion_seg_just_CaSupp_100", 1662491259, "189199d231068b4766f829cfd7132d58", ("CaSupp_100", "mask_spine")),
    (717, "MM_Lesion_seg_all_VMI", 1662872416, "af18e90a1fe5933e4a596d6463f02139", ("VMI_40", "VMI_80", "VMI_120", "mask_spine")),
    (718, "MM_Lesion_seg_all_CaSupp", 1663320518, "6e71269971b4826e968775335e03b372", ("CaSupp_25", "CaSupp_50", "CaSupp_75", "CaSupp_100", "mask_spine")),
)

LESION_MODELS = MappingProxyType({
    model_id: ModelSpec(
        model_id, "Dataset{}_{}".format(model_id, suffix),
        "Dataset{}_{}.zip".format(model_id, suffix), 19708429, size, checksum,
        channels, channels[:-1], (0, 1, 2, 3, 4),
    )
    for model_id, suffix, size, checksum, channels in _ROWS
})

SPINE_MODEL = ModelSpec(
    dataset_id=802,
    dataset_name="Dataset802_Spine_segmentation_trained_on_VerSe20_and_MM_dataset_together",
    zip_filename="Spinal-Multiple-Myeloma-SEG_nnUNet_models.zip",
    zenodo_record_id=18598645,
    zip_size=3681185990,
    md5="02f72a4c4bf8c34790b36400db7cc6d1",
    channel_names=("konvCT",),
    required_modalities=("ConvCT",),
    folds=("all",),
    expected_nnunet_version=None,  # Exact training version not verified.
    binary_mask_channel=None,
    archive_root="Spinal-Multiple-Myeloma-SEG_nnUNet_models/",
)


def get_lesion_model(dataset_id: int) -> ModelSpec:
    """Return immutable metadata; raise ValueError for unsupported lesion IDs."""
    try:
        return LESION_MODELS[dataset_id]
    except KeyError:
        raise ValueError("Unsupported lesion model {}; expected 700-718".format(dataset_id)) from None


def get_model(dataset_id: int) -> ModelSpec:
    """Look up either a benchmark lesion model or the separate spine model."""
    return SPINE_MODEL if dataset_id == 802 else get_lesion_model(dataset_id)
