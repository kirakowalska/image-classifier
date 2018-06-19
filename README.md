# image-classifier
Basic hand-written digit and letter classifier using CNN (convolutional neural networks) implemented in Tensorflow. 

The classifier has the following architecture:

![alt text](https://github.com/kirakowalska/image-classifier/blob/master/cnn_architecture.png)

## Data preparation

The data contains 30,134 images of handwritten letters and digits. The images are of size 56x56 binary pixels and are stored as 3136-dimensional rows in the input matrix train_x. The labels are coded with integers from 0 to 35 and stored in the input vector train_y. The file train.pkl contains a pickled tuple (train_x, train_y).

We split the data into three data sets:
* Training images: 18,082 (60%)
* Validation images: 6,026 (20%)
* Test images: 6,026 (20%)

## Performance

Test accuracy: 91.93%

Test accuracy with data augmentation during training: 92.57%

Test accuracy is the expected accuracy of the classifier. It is estimated by running the classifier on 20% of data that was withheld from training.

## Technologies used: 
* TensorFlow for deep learning 
* Numpy, Sci-kit for numerical manipulations
* Imgaug for image augmentation

## Techniques used for improving the classifier:

* **Early stopping**:

The training of the classifier was terminated after 23 epochs to avoid overfitting. Empirically, we noticed that if we left the classifier to train for longer, it's performance on the training data would have continued to increase to 100% accuracy, but it would have performed worse on validation data due to overfitting. 

* **Data augmentation**: 

Training data was augmented by cropping, scaling and rotating in order to enrich the training data and as a result make the classifier perform better on unseen images. The augmentation resulted in increased accuracy on test images (see performance comparison above). It also reduced the problem of overfitting on training data and hence removed the need for 'early stopping' during training.

* **Potential extentions** 

If time permitted, other techniques for improving the classifier could be investigated, such as hyperparameter tuning or regularisation (e.g. dropout).
