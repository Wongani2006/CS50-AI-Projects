# Traffic Sign Classifier

This project classifies German traffic sign images into 43 categories using TensorFlow and a convolutional neural network. The program loads image files from numbered folders (`0` through `42`), resizes each image to 30 × 30 pixels, and uses the folder number as its label. It splits the data into training and test sets, trains the model, and reports the test loss and accuracy.

## Model and experiment

I implemented a CNN with two convolution and max-pooling stages (32 and 64 filters), followed by a 128-unit dense layer, dropout, and a 43-unit softmax output. Pixel values are scaled to the range 0–1, and the model is trained for 10 epochs with Adam and categorical cross-entropy. In a full run on the supplied GTSRB dataset, the model reached **98.15% accuracy** and **0.0881 loss** on the held-out 40% test split. This showed that the CNN learned the sign patterns well. The training accuracy reported during the final epoch was lower than test accuracy, in part because dropout is active during training and disabled during evaluation. The split is random, so a later run can produce a somewhat different result. I did not compare multiple architectures or tune hyperparameters, so those are possible next experiments.

## Run

Install the dependencies listed in `requirements.txt`, then run:

```bash
python traffic.py gtsrb
```

The `gtsrb` directory must contain the 43 numbered category folders. To save the trained model, provide an output filename:

```bash
python traffic.py gtsrb model.keras
```
