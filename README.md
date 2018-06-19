# image-classifier
Basic hand-written digit and letter classifier using CNN (convolutional neural networks) implemented in Tensorflow.

The classifier has the following architecture:

![alt text](https://github.com/kirakowalska/image-classifier/blob/master/cnn_architecture.png)

## Performance
Training accuracy: 96.9%

Validation accuracy: 96.9%

Validation accuracy is the expected accuracy of the classifier on unseen data. It is estimated by running the classifier on 20% of data that was withheld from training.

## Technologies used: 
* TensorFlow for deep learning 
* Numpy, Sci-kit for numerical manipulations
* Imgaug for image augmentation

## Techniques used for improving the classifier:
* **Data augmentation**: 

Training data was augmented by cropping, scaling and rotating in order to enrich the data and as a result make the classifier perform better on unseen images. The augmentation is not used in the final version of the model as there is enough training data available (before augmentation) to achieve near 100% accuracy on unseen images.

* **Early stopping**:

The training of the classifier was terminated early to avoid overfitting. Empirically, the optimal training duration was determined to be ~12 epochs. If we left the classifier to train for longer, it's performance on the training data would have continued to increase to 100% accuracy, but it would have performed worse on unseed data due to overfitting.

* **Potential extentions** 

Other techniques for improving the classifier could be investigated, such as hyperparameter tuning or regularisation (e.g. dropout), but they were not needed in this project due to simplicity of the classification problem at hand.
