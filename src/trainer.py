import tensorflow as tf
from loss import Loss

class Trainer:

    def __init__(self, is_fp16, loss_config, model, opt):
        self.loss_fn = Loss(loss_config)
        if is_fp16:
            self.train_fn = self.train_fn_fp16
        else:
            self.train_fn = self.train_fn_fp32
        self.model = model
        self.opt = opt
        
    @tf.function
    def train_fn_fp16(inputs, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(inputs, training=True)
            loss = self.loss_fn(y_true, y_pred)
            scaled_loss = self.opt.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.opt.get_unscaled_gradients(scaled_gradients)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred

    @tf.function
    def train_fn_fp32(inputs, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(img, training=True)
            loss = self.loss_fn(y_true, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred

    @tf.function
    def val_fn(inputs, y_true):
        y_pred = model(img, training=False)
        loss = self.loss_fn(y_true, y_pred)
        return loss, y_pred




