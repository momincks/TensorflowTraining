import tensorflow as tf
from .manager import SingleLossManager

class CharbonnierLoss(SingleLossManager):

    def __init__(self, cfg):
        super(CharbonnierLoss).__init__(self)
        self.sq_epsilon = cfg["squared_epsilon"]

    @tf.function
    def calc(self, x, y):
        return tf.reduce_sum(tf.math.sqrt(tf.math.square(x - y) + self.sq_epsilon))

class SobelLoss(SingleLossManager):

    def __init__(self, cfg):
        super(SobelLoss).__init__(self)
        self.mode == cfg["mode"]
        if self.mode == "charbonnier":
            self.charbonnier = CharbonnierLoss(cfg["charbonnier"]["squared_epsilon"])

    @tf.function
    def sobel_charbonnier(self, x, y):
        return self.charbonnier(tf.image.sobel_edges(x), tf.image.sobel_edges(y))

    @tf.function
    def sobel_L1(self, x, y):
        return tf.reduce_sum(tf.math.subtract(tf.image.sobel_edges(x), tf.image.sobel_edges(y)))

    def calc(self, x, y):

        if self.mode == "charbonnier":
            return self.sobel_charbonnier(x, y)

        elif self.mode == "L1":
            return self.sobel_L1(x, y)

        else:
            raise ValueError("wrong sobel loss mode!")