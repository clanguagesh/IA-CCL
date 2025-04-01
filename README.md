# IACCL

## Model Download

You can download the model weight from the following link:
[dota15_10_result.pth](https://drive.google.com/uc?export=download&id=1VO96NUfb6uMsEKxzqGaMpvJUZGddHnFf)
[dota15_20_result.pth](https://drive.google.com/uc?export=download&id=1dDqjhGEUNYCvaGuDANbWyKWa47SsSVYR)

# Train
The training code will be available when my paper is accepted.

# Test
Before test IACCL, you need replace the data-path in configs/_base_/datasets/dotav15.py, then you need download the pth and put it in checkpoint directory. You could test IACCL with the following command:
```
python test.py configs/ssad_fcos/base_IACCL_default.py checkpoint/dota15_1
0_result.pth --eval mAP
```
You could visulize an image with the following command:
```
python image_demo.py demo.png configs/ssad_fcos/base_IACCL_default.py checkpoint/dota15_20_result.pth --out-file visualize.png
```
