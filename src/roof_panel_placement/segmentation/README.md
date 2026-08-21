# Segmentation package

This package is populated one experimental stage at a time.

The first stage is roof segmentation. It will contain three strictly separate
implementations:

- multiscale SLIC with hand-crafted features and a Random Forest;
- a task-trained U-Net; and
- a prompted pipeline using an object detector to prompt a segmentation model.

All implementations return binary roof masks so that evaluation and
reporting do not depend on the method. Oracle masks supervise U-Net fitting and
score development results, but no deployable inference function accepts them.

- `slic_rf.py` implements multiscale SLIC features and a Random Forest.
- `unet.py` implements a compact task-trained U-Net and its training loop.
- `grounded_sam.py` implements joint, independent, consensus, parent-box, and
  hierarchical Grounding DINO + SAM 2.1 prompting. The clipped hierarchy uses
  parent-building SAM regions, while oracle-component boxes are exposed only
  through an explicitly diagnostic function. V3 additionally exposes parent
  boxes so parent coverage and final roof-box localization can be evaluated as
  separate stages.
