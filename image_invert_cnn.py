import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras import backend as K

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def show_weights_histogram(model, name='block1_conv1'):
    w = model.get_layer(name).get_weights()[0]
    plt.hist(w.flatten())
    plt.title("w values after 'imagenet' weights are loaded")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_image_path', type=str, default='./content_image.jpg', help='-')
    parser.add_argument('--image_resize', type=int, default=(224, 224), help='-')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--training_epoch', type=int, default=10000, help='-')
    parser.add_argument('--tv_lambda', type=float, default=1e-7, help='-')

    args, unknown = parser.parse_known_args()

    """ HYPER PARAMETER  """
    content_img_path = args.content_image_path
    image_resize = args.image_resize
    learning_rate = args.learning_rate
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    epsilon = args.epsilon
    epoch = args.training_epoch
    tv_lambda = args.tv_lambda


    # generated_image is trainable parameter. Initialize by random_normal noise.
    content_image = cv2.imread(content_img_path).astype(np.float32)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    content_image = cv2.resize(content_image, image_resize, interpolation=cv2.INTER_CUBIC)
    generated_image = tf.Variable(1e-1 * tf.random_normal(shape=content_image.shape), dtype=tf.float32, name='random_noise', trainable=True)


    # preprocessing - subtract mean rgb value of 'imagenet'
    content_image[:, :, 0] -= 123.68
    content_image[:, :, 1] -= 116.779
    content_image[:, :, 2] -= 103.939


    # Reshape to 1 batch size.
    image_shape = (1,) + content_image.shape  # shape = (1,224,224,3)
    content_image = tf.reshape(tf.constant(content_image, dtype=tf.float32), shape=image_shape)
    generated_image = tf.reshape(generated_image, shape=image_shape)


    # Concatenate with two image(Tensors).
    input_tensor = tf.concat([generated_image, content_image], axis=0)


    # Load pretrained model using Keras API
    with tf.variable_scope('pretrained_model'):
        model = VGG19(weights='imagenet', input_tensor=input_tensor, include_top=False)
        keras_variables = [var.name for var in tf.global_variables() if 'pretrained_model' in var.name]

    # Output Tensor of Keras model into Dictionary
    output_dict = {layer.name: layer.output for layer in model.layers}


    # Loss
    feature_vectors = output_dict['block4_conv1']
    generated_feature = feature_vectors[0, :, :, :]
    content_feature = feature_vectors[1, :, :, :]

    """
    (l2_normalize)||Φ(σx) − Φ0||/||Φ0||   +    RV(x)
    """
    l2_loss = tf.norm(content_feature - generated_feature, 'euclidean') / tf.norm(content_feature, 'euclidean')
    total_variation_loss = tv_lambda * tf.reduce_sum(tf.image.total_variation(generated_image))
    loss = l2_loss + total_variation_loss


    # Minimize cost
    trainble_variables = [var for var in tf.global_variables() if 'pretrained_model' not in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2, epsilon=epsilon).minimize(loss, var_list=trainble_variables)


    # Session
    sess = K.get_session()
    uninitialize_variables = [var for var in tf.global_variables() if var.name not in keras_variables]
    sess.run(tf.variables_initializer(uninitialize_variables))
    K.set_session(sess)

    # model.summary()
    # Check to the weights of pretrained model is not initialized.
    # show_weights_histogram(model)
    # model.trainable = False

    with sess.as_default():
        avg_cost = 0

        for i in range(epoch):
            feat_val, _cost, _ = sess.run([feature_vectors, loss, optimizer])
            avg_cost += _cost / epoch

            if i % 100 == 0:
                print(i, _cost)
                # show_weights_histogram(model)
                # print(feat_val[1,:,:,:])

    print("Complete Reconstruct Image!!")

    reconstructed_image = sess.run(generated_image)[0]

    # deprocessing - add mean rgb value of 'imagenet'
    reconstructed_image[:, :, 0] += 123.68
    reconstructed_image[:, :, 1] += 116.779
    reconstructed_image[:, :, 2] += 103.939
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype('uint8')
    reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)

    K.clear_session()

    cv2.imwrite('./reconstructed_image.jpg', reconstructed_image)