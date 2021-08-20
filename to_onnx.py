import numpy as np
import tensorflow as tf
import tf2onnx
from train import get_model

size_h = 160
size_w = 160
channel = 3
num_class = 3

model = get_model(size_h=size_h, size_w=size_w, num_class=num_class, load_path="./weights/360helmet4/15000_0.0151.h5")
spec = (tf.TensorSpec((None, size_h, size_w, channel), tf.float32, name="input"),)

model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model,
                input_signature=spec, opset=None, custom_ops=None,
                custom_op_handlers=None, custom_rewriter=None,
                inputs_as_nchw=None, extra_opset=None, shape_override=None,
                target=None, large_model=False, output_path="./output.onnx")