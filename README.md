# leaf-plant-classification
Deep Learning models to classify plant leafs

### Report

#### Dataset

The given dataset was clearly unbalanced, with a class with a huge number of samples (Tomato > 5000 samples).
We trained the first model on the untouched dataset, to review the initial performance, but we got low accuracy results (as expected).
We considered three possible solutions:
  1. Introducing class weights
  2. Undersampling the dataset
  3. Oversampling the dataset
Oversampling didn't seem like a good idea, because of the image augmentation we're performing later, and the great discrepancy between class sizes. Also, the training time would increase.

The dataset was split into three subsets with the following sizes: training (70%), validation (20%) and testing (10%, even if not required).
The training subset is kept bigger to have sufficient samples to train a well-performing model, while validation and testing subsets are kept smaller; in particular, the testing subset was used to better analyze the model performance.

![hann1](./francescoari/hann1.png "Img 1")

#### Feature Engineering

