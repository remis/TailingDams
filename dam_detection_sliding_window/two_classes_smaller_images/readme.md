Applying the neural net trained from scratch in dam_recognition.two_classes_cv_from_scratch in a sliding window manner to an image of a bigger area for dam detection

Architecture is LeNet5-style

First choosing the best performed neural network among nets trained on 10 splits on train (1200 images) and test (300 images) datasets.

2-class classification problem: dams, not dams = {not dams, lookalike}

Smaller images 95x95

