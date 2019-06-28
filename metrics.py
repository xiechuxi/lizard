import numpy as np
import tensorflow as tf
from keras import backend as K
import keras

# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    #return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
    return keras.losses.binary_crossentropy(y_true, y_pred)
    #return 1.0 - dice_coef(y_true, y_pred)

"""
"It is well known that in order to get better results your evaluation metric and your loss function need to be as similar as possible.
The problem here however is that Jaccard Index is not differentiable. One can generalize it for probability prediction, which on one hand, 
in the limit of the very confident predictions, turns into normal Jaccard and on the other hand is differentiable - allowing the usage of 
it in the algorithms that are optimized with gradient descent."
(http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/)
"""
def custom_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) - K.log(mean_iou(y_true, y_pred))