import sys, os, math ,time, gc, glob, random, yaml, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from shutil import move, copy2
from collections import deque
from datetime import datetime
from easydict import EasyDict

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

from src.model.efficientnetv2 import effnetv2_model
from src.dataloader import ImageLoader, TFRecordLoader
from src.trainer import Trainer

def init():
    gc.enable()

    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="./config.yml", help="path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        cfg = EasyDict(yaml.safe_load(file))
    return cfg
        
def get_model(size_h, size_w):
    model = effnetv2_model.get_model('efficientnetv2-b0', include_top=False, pretrained=True)
    inputs = tf.keras.Input([size_h, size_w, 3], name="input")
    x = inputs
    x = model(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(num_class, activation='softmax', dtype=tf.float32, name="output")(x)
    outputs = x
    model = tf.keras.Model(inputs, outputs, name="model")
    print(model.summary())
    return model

if __name__=="__main__":

    cfg = init()

    train_cfg = cfg["training"]
    lr_cfg = cfg["scheduler"]
    loss_cfg = cfg['lossfn']
    opt_cfg = cfg['optimizer']
    data_cfg = cfg["dataset"]

    size_h = train_cfg["size_height"]
    size_w = train_cfg["size_width"]
    lr_init = train_cfg["learning_rate"]
    batch_size = train_cfg["batch_size"]
    weight_decay = train_cfg["weight_decay"]

    dir_map = data_cfg["directory_map"]
    val_split = data_cfg["validation_split"]
    class_name = data_cfg["class_name"]
    num_class = len(class_name)

    if data_cfg["using_tfrecord"]:
        ds = TFRecordLoader(size_h=size_h,
                            size_w=size_w,
                            batch_size=batch_size,
                            val_split=val_split,
                            num_class=num_class,
                            class_name=class_name,
                            dir_map=dir_map
                            )
    else:
        ds = ImageLoader(size_h=size_h,
                            size_w=size_w,
                            batch_size=batch_size,
                            val_split=val_split,
                            num_class=num_class,
                            class_name=class_name,
                            dir_map=dir_map
                        )

    train_ds = ds.get("train")
    val_ds = ds.get("val")
    train_num, val_num, train_info, val_info = ds.get("info")

    if opt_cfg["using"] == "sgdw":
        pass
    elif opt_cfg["using"] == "adamw":
        pass
    elif opt_cfg["using"] == "novograd":
        opt = tfa.optimizers.NovoGrad(weight_decay=opt_cfg["novograd"]["weight_decay"], epsilon=opt_cfg["novograd"]["epsilon"])

    if train_cfg["using_float16"]:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, loss_scale="dynamic")

    model = get_model(size_h, size_w)
    using_fp16 = train_cfg["using_float16"]
    trainer = Trainer(using_fp16, loss_config, model, opt)

    validate_in = train_cfg["validate_in"]
    num_batch = train_cfg["num_batch"]
    save_path = train_cfg["save_path"]

    if lr_cfg["using"] == "cosine_decay_restarts":
        lr_params = lr_cfg["cosine_decay_restarts"]
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr_params["lr_init"],
                                                                            first_decay_steps=lr_params["first_decay_steps"],
                                                                            t_mul=lr_params["t_mul"],
                                                                            m_mul=lr_params["m_mul"],
                                                                            alpha=lr_params["alpha"],
                                                                        )
    else:
        pass

    train_metrics = [tf.keras.metrics.CategoricalAccuracy()]
    val_metrics = [tf.keras.metrics.CategoricalAccuracy()]
    for class_id in range(num_class):
        train_metrics.extend([tf.keras.metrics.Precision(class_id), tf.keras.metrics.Recall(class_id)])
        val_metrics.extend([tf.keras.metrics.Precision(class_id), tf.keras.metrics.Recall(class_id)])

    while True:

        train_losses = deque([], 500)
        for train_batch_id, (inputs, y_true) in enumerate(train_ds, 1):
            train_loss, y_pred = trainer.train_fn(inputs, y_true)
            train_losses.append(tf.reduce_mean(train_loss))

            y_true = tf.one_hot(tf.math.argmax(y_true, axis=1), num_class)
            for i in train_metrics:
                i.update_state(y_true, y_pred)      
            scores = [i.result().numpy().round(3) for i in train_metrics]

            old_lr = tf.keras.backend.get_value(opt.learning_rate).round(4)
            new_lr = linear_warmup_step_down(batch_count)
            if new_lr != old_lr:
                tf.keras.backend.set_value(opt.learning_rate, new_lr)

            show_loss = round(sum(train_losses)/len(train_losses), 3)
            print(f"train... batch: {batch_id} lr: {old_lr} loss: {show_loss} metric: {scores}", end="\r")

            if train_batch_id in validate_in:
                print("\nvalidation begins...")
                val_losses = []
                for val_batch_id, (inputs, y_true) in enumerate(val_ds, 1):
                    val_loss, y_pred = trainer.val_fn(inputs, y_true)
                    val_losses.append(tf.reduce_mean(val_loss))
                    y_true = tf.one_hot(tf.math.argmax(y_true, axis=1), num_class)
                    for i in val_metrics:
                        i.update_state(y_true, y_pred)      
                    scores = [i.result().numpy().round(3) for i in val_metrics]
                    show_loss = round(sum(val_losses)/len(train_losses), 3)
                    print(f"val... batch: {val_batch_id} loss: {show_loss} metric: {scores}", end="\r")
                [i.reset_states() for i in train_metrics]
                [i.reset_states() for i in val_metrics]
                os.makedirs(save_path, exist_ok=True)
                model.save_weights(f"{train_batch_id}_{show_loss}.h5")
                log = open("logging.txt", "a+")
                log.write(f"{str(datetime.now())}: {train_batch_id} loss: {show_loss} metrics: {scores}")
                log.close()
                print("\nvalidation ended...")

        if train_batch_id == num_batch:
            print("training completed...")
            break