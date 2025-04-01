# IACCL

# Train
The training code will be available when my paper is accepted.

# Test
You could test IACCL with the following command, before run this command, you need replace the data-path in configs/_base_/datasets/dotav15.py:
```
python test.py configs/ssad_fcos/base_IACCL_default.py checkpoint/dota15_1
0_result.pth --eval mAP
```
You could visulize an image with the following command:
```
python image_demo.py demo.png configs/ssad_fcos/base_IACCL_default.py checkpoint/dota15_20_result.pth --out-file visualize.png
```