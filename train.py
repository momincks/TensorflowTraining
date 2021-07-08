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
from src.trainer import Trainer, LRScheduler

def init():
    gc.enable()

    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="path to config file")
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
    batch_size = train_cfg["batch_size"]

    dir_map = data_cfg["directory_map"]
    val_split = data_cfg["validation_split"]
    class_name = data_cfg["class_name"]
    wrong_label = data_cfg["wrong_label"]

    num_class = len(class_name)
    if data_cfg["using_tfrecord"]:
        ds = TFRecordLoader(train_cfg=train_cfg,
                            data_cfg=data_cfg,
                            size_h=size_h,
                            size_w=size_w,
                            batch_size=batch_size,
                            val_split=val_split,
                            shuffle=8192,
                            classe_name=class_name,
                            wrong_label=wrong_label,
                            dir_map=dir_map
                            )
    else:
        ds = ImageLoader(train_cfg=train_cfg,
                            data_cfg=data_cfg,
                            size_h=size_h,
                            size_w=size_w,
                            batch_size=batch_size,
                            val_split=val_split,
                            shuffle=8192,
                            class_name=class_name,
                            wrong_label=wrong_label,
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
    trainer = Trainer(using_fp16, loss_cfg, model, opt)

    validate_in = train_cfg["validate_in"]
    reshape_in = train_cfg["reshape_in"]
    more_intense_in = train_cfg["more_intense_in"]
    training_size = train_cfg["training_size"]
    num_batch = train_cfg["num_batch"]
    save_path = train_cfg["save_path"]

    lr_scheduler = LRScheduler(lr_cfg)

    train_metrics = [tf.keras.metrics.CategoricalAccuracy()]
    val_metrics = [tf.keras.metrics.CategoricalAccuracy()]
    for class_id in range(num_class):
        train_metrics.extend([tf.keras.metrics.Precision(class_id=class_id), tf.keras.metrics.Recall(class_id=class_id)])
        val_metrics.extend([tf.keras.metrics.Precision(class_id=class_id), tf.keras.metrics.Recall(class_id=class_id)])


    train_batch_id = 0
    while True:

        train_losses = deque([], 100)
        for (inputs, y_true) in train_ds:

            train_batch_id += 1
            train_loss, y_pred = trainer.train_fn(inputs, y_true)
            train_losses.append(tf.reduce_mean(train_loss).numpy())

            y_true = tf.one_hot(tf.math.argmax(y_true, axis=1), num_class)
            for i in train_metrics:
                i.update_state(y_true, y_pred)

            old_lr = tf.keras.backend.get_value(opt.learning_rate)
            new_lr = lr_scheduler(train_batch_id)
            tf.keras.backend.set_value(opt.learning_rate, new_lr)

            show_loss = "{:.4f}".format(sum(train_losses)/len(train_losses))
            show_lr = "{:.5f}".format(old_lr)
            show_scores = ["{:.4f}".format(i.result().numpy()) for i in train_metrics]
            sys.stdout.write("\x1b[2K")
            print(f"train ... batch: {train_batch_id} lr: {show_lr} loss: {show_loss} metric: {show_scores}", end="\r")

            if train_batch_id in validate_in:
                print(f"\nvalidation begins at {train_batch_id} ...")
                val_losses = []
                for val_batch_id, (inputs, y_true) in enumerate(val_ds, 1):
                    val_loss, y_pred = trainer.val_fn(inputs, y_true)
                    val_losses.append(tf.reduce_mean(val_loss).numpy())
                    y_true = tf.one_hot(tf.math.argmax(y_true, axis=1), num_class)
                    for i in val_metrics:
                        i.update_state(y_true, y_pred)      
                    show_loss = "{:.4f}".format(sum(val_losses)/len(val_losses))
                    show_scores = ["{:.4f}".format(i.result().numpy()) for i in val_metrics]
                    print(f"val ... batch: {val_batch_id} loss: {show_loss} metric: {show_scores}", end="\r")
                [i.reset_states() for i in train_metrics]
                [i.reset_states() for i in val_metrics]
                os.makedirs(save_path, exist_ok=True)
                weight_name = f"{train_batch_id}_{show_loss}.h5"
                model.save_weights(os.path.join(save_path, weight_name))
                log_name = os.path.join(save_path, "logging.txt")
                log = open(log_name, "a+")
                log.write(f"{str(datetime.now())}: {train_batch_id} loss: {show_loss} metrics: {show_scores}")
                log.close()
                print("\nvalidation ended ...")

            if train_batch_id in more_intense_in:
                ds.more_intense()
                print("\ntriggered more intense augmentation at {train_batch_id} ...")

            ### debug section ###
            if train_batch_id in [10,20,30]:
                img = tf.cast(tf.clip_by_value(inputs[0]*255.,0,255.),tf.uint8)
                img = tf.io.encode_jpeg(img[0], format='', quality=90, progressive=False, optimize_size=True, chroma_downsampling=False)
                tf.io.write_file(f'tmp_{train_batch_id}.jpg',img)

            # if train_batch_id in reshape_in:
            #     current_height, current_width = training_size[0]["height"], training_size[0]["width"]
            #     model = get_model(current_height, current_width)
            #     ds.bigger_size(current_height, current_width)
            #     del training_size[0]
            #     print(f"\nCurrent Height: {current_height}, Current Width: {current_width}\n")

            if train_batch_id >= num_batch:
                print("\ntraining completed ...")
                break

        if train_batch_id >= num_batch:
            break