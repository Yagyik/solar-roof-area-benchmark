# Segmentation package

This package is populated one experimental stage at a time.

The first stage is roof segmentation. It will contain three strictly separate
implementations:

- multiscale SLIC with hand-crafted features and a Random Forest;
- a task-trained U-Net; and
- a prompted pipeline using an object detector to prompt a segmentation model.

Both implementations will return the same result schema so that evaluation and
reporting do not depend on the method. Oracle roof masks will be read only by
the evaluation layer, never by either deployable implementation.

`slic_rf.py` contains the first classical implementation. The learned and
prompted implementations will be added only after the classical notebook has
run successfully in Colab.

