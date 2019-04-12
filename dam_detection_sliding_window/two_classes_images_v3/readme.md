Applying the neural net trained from scratch in dam_recognition.two_classes_images_v3 in a sliding window manner to an image of a bigger area for dam detection

Architecture is LeNet5-style

2-class classification problem: dams, not dams = {not dams, lookalike}

Version 3 of dam images - 134x134, smaller regions, bigger dams, removed dams in urban areas.
Version 2 of not a dam images - 134x134, middle regions of the original images plus new images over false positive results from previous training.
Original look-a-like images - cropped to 134x134

Data augmentation is applied. For each of train image 3 augmented images are generated. Augmentation is done as:
1. random horizonal flip - with a probability 0.5 the image is flipped horizontally
2. random vertical flip - with a probability 0.5 the image is flipped vertically
3. random rotation - with a probability 0.5 the image is rotated by 90, 180, or 270 degree

