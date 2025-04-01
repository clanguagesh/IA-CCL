from configs.rotated_fcos.rotated_fcos_r50_fpn_le90 import model as detector
from configs._base_.datasets.dotav15 import data as src_data

_base_ = [
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]


custom_imports = dict(
    imports=['ssad'],
    allow_failed_imports=False)

data = dict(
    val=src_data['val'],
    test=src_data['test'],
)

detector['bbox_head']['num_classes'] = 16
model = dict(
    type="RotatedSemiIACCL",
    model=detector,
    test_cfg=dict(inference_on="teacher"),
)
