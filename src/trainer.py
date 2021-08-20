import tensorflow as tf

class LRScheduler:

    def __init__(self, lr_cfg):
        self.using = lr_cfg["using"]
        if self.using == "warmup_cosine_decay_restarts":
            lr_params = lr_cfg["warmup_cosine_decay_restarts"]
            self.scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr_params["lr_init"],
                                                                            first_decay_steps=lr_params["first_decay_steps"],
                                                                            t_mul=lr_params["t_mul"],
                                                                            m_mul=lr_params["m_mul"],
                                                                            alpha=lr_params["alpha"],
                                                                            )
            self.warmup_steps = lr_params["warmup_steps"]
            self.warmup_from = lr_params["warmup_from"]
            self.warmup_inteval = (lr_params["lr_init"] - self.warmup_from) / self.warmup_steps

    def __call__(self, batch_id):
        if self.using == "warmup_cosine_decay_restarts":
            if self.warmup_steps:
                if batch_id < self.warmup_steps:
                    return self.warmup_from + self.warmup_inteval * batch_id
                else:
                    return self.scheduler(batch_id - self.warmup_steps)

class LossManager:
    
    def __init__(self, loss_config):
        self.bce_epsilon, = loss_config["bce"]
        self.focal_alpha = loss_config["focal"]["alpha"]
        self.focal_gamma = loss_config["focal"]["gamma"]
        self.focal_epsilon = loss_config["focal"]["epsilon"]
        if loss_config["using"] == "bce":
            self.loss_fn = self.bce
        elif loss_config["using"] == "focal":
            self.loss_fn = self.focal

    def bce(self, y_true, y_pred):
        return - y_true * tf.math.log(y_pred + self.bce_epsilon) - (1. - y_true) * tf.math.log(1. - y_pred + self.bce_epsilon)

    def focal(self, y_true, y_pred):
        alpha_t = y_true * self.focal_alpha + (tf.ones_like(y_true) - y_true) * (1. - self.focal_alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1. - y_true, 1. - y_pred)
        fl = tf.multiply(tf.multiply(tf.math.pow(tf.subtract(1., y_t), self.focal_gamma), - tf.math.log(y_t + self.focal_epsilon)), alpha_t)
        return tf.reduce_mean(fl)

    def get(self):
        return self.loss_fn

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

class Trainer:

    def __init__(self, is_fp16, loss_config, model, opt):
        self.loss_fn = LossManager(loss_config).get()
        if is_fp16:
            self.train_fn = self.train_fn_fp16
        else:
            self.train_fn = self.train_fn_fp32
        self.model = model
        self.opt = opt
        
    @tf.function
    def train_fn_fp16(self, inputs, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(inputs, training=True)
            loss = self.loss_fn(y_true, y_pred)
            scaled_loss = self.opt.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.opt.get_unscaled_gradients(scaled_gradients)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred

    @tf.function
    def train_fn_fp32(self, inputs, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(inputs, training=True)
            loss = self.loss_fn(y_true, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred

    @tf.function
    def val_fn(self, inputs, y_true):
        y_pred = self.model(inputs, training=False)
        loss = self.loss_fn(y_true, y_pred)
        return loss, y_pred
