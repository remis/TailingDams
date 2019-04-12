Training a neural net from scratch

Architecture is LeNet5-style

10 splits on train (1355 images) and test (300 images) datasets.

2-class classification problem: dams, not dams = {not dams, lookalike}

For dams the version 3 images are used, 134x134 image patches with removed dams in urban areas

For not dams the version 2 images are used

For lookalike a smaller image patch is cropped from the middle of an original image

Data augmentation is applied. For each of train image 3 augmented images are generated. Augmentation is done as:
1. random horizonal flip - with a probability 0.5 the image is flipped horizontally
2. random vertical flip - with a probability 0.5 the image is flipped vertically
3. random rotation - with a probability 0.5 the image is rotated by 90, 180, or 270 degree

