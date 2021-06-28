import tensorflow as tf

class Loss:
    
    def __init__(self, loss_config):
        self.bce_epsilon, = loss_config["bce"]
        self.focal_alpha, self.focal_gamma, self.focal_epsilon = loss_config["focal"]
        if loss_config["using"] == "bce":
            self.loss_fn = self.bce
        elif loss_config["using"] == "focal":
            self.loss_fn = self.focal

    def bce(y_true, y_pred):
        return - y_true * tf.math.log(y_pred + self.bce_epsilon) - (1 - y_true) * tf.math.log(1 - y_pred + self.bce_epsilon)

    def focal(y_true, y_pred):
        alpha_t = y_true * self.focal_alpha + (tf.ones_like(y_true) - y_true) * (1. - self.focal_alpha)
        y_t = tf.multiply(y_true, y_pred + self.focal_epsilon) + tf.multiply(1. - y_true, 1. - y_pred + self.focal_epsilon)
        fl = tf.multiply(tf.multiply(tf.math.pow(tf.subtract(1., y_t), self.focal_gamma), - tf.math.log(y_t)), alpha_t)
        return tf.reduce_mean(fl)

    def __call__(self, *args):
        return self.loss_fn(*args)

    # def PC(y_true, y_pred):
    #     y_true_left = y_true[0:batch_size//2,:]
    #     y_true_right = y_true[batch_size//2:,:]
    #     y_pred_left = y_pred[0:batch_size//2,:]
    #     y_pred_right = y_pred[batch_size//2:,:]
    #     mask = tf.equal(tf.argmax(y_true_left,-1),tf.argmax(y_true_right,-1))
    #     mask = not mask
    #     mask = tf.cast(mask,tf.float32)
    #     loss = tf.abs(y_pred_left-y_pred_right)
    #     loss = tf.reduce_sum(loss,-1) * mask
    #     loss = tf.reduce_sum(loss) / batch_size
    #     return loss

    # def bitemper(y_true, y_pred):
    #     loss = bitemperloss.bi_tempered_logistic_loss(y_pred,y_true,bitemper_t1,bitemper_t2,0.0,5)
    #     return loss
