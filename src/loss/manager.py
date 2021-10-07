
import tensorflow as tf
from . import utils

class SingleLossManager:

    def __init__(self):
        pass

    def calc(self):
        pass

    def __call__(self, *args):
        return self.calc(args)

class MultiLossManager:

    def __init__(self, loss_name_string):
        self.loss_cls = []
        loss_names = utils.ParseMultiLoss(loss_name_string)
        for loss_name in loss_names:
            loss_mod = utils.ImportLoss(loss_name)
            self.loss_cls.append(getattr(loss_mod, loss_name))
        
    def __call__(self):
        return tf.reduce_sum([i.calc() for i in self.loss_cls])