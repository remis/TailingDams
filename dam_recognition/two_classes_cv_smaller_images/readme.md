Training a neural net from scratch

Architecture is LeNet5-style

10 splits on train (790 images) and test (100 images) datasets.

2-class classification problem: dams, not dams = {not dams, lookalike}

For dams the version 2 images are used, smaller images and bigger dams, for not dams images are cropped from original images of the same size:
for lookalike a smaller image patch is cropped from the middle of an original image, for no dams two random image patches are sampled from each original image

