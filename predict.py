import tensorflow as tf
import numpy as np
import os,glob,cv2
# import sys,argparse
import pickle

def predict(input_data):

    image_size = 56
    num_channels = 1
    classes = range(0, 36)
    num_classes = len(classes)

    # Format test data
    print('Going to read test images')
    x_test = []
    for img in input_data:
      image = img.reshape((image_size, image_size,num_channels))
      x_test.append(image)
    image_no = len(x_test)
    x_test = np.array(x_test)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('image-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred_cls = graph.get_tensor_by_name("y_pred_cls:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((image_no, num_classes))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_test, y_true: y_test_images}
    output_data=sess.run(y_pred_cls, feed_dict=feed_dict_testing)
    # result is an array of predicted labels
    return output_data

if __name__ == "__main__":
    train_path = ''
    train_file = os.path.join(train_path, 'train.pkl')
    raw_images, raw_labels = pickle.load(open(train_file, 'rb'))[0:100]

    predicted_labels = predict(raw_images)
    print(predicted_labels.shape)