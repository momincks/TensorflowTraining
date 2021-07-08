import os, math, pathlib, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from glob import glob
from tensorflow.data import Dataset, TFRecordDataset
from tensorflow.keras.preprocessing import image_dataset_from_directory

class Loader:

    # loading images or tfrecord as training data
    # use dir_map to map directories to class_name
    # map to label "frame" to random crop it and train it as background

    def __init__(self, train_cfg, data_cfg, size_h, size_w, batch_size, val_split, shuffle, class_name, wrong_label, dir_map,
                augment_multiple, augment_step, min_intensity):
        self.size_h, self.size_w = size_h, size_w
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_name = class_name
        self.num_class = len(class_name)
        self.wrong_label = tf.constant(wrong_label)
        self.on_value = 0.95
        self.off_value = (1. - self.on_value)/(self.num_class-1)
        self.cache_image = None
        self.cache_label = None
        self.augment_multiple = augment_multiple
        self.augment_step = augment_step
        self.intensity = min_intensity.copy()
        self.min_intensity = min_intensity
        self.max_intensity = {k:v*augment_multiple for k,v in self.min_intensity.items()}
        self.training_size = train_cfg["training_size"]
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def single_augment(self,img,label):   
    # augment for single image       
        img = tfa.image.shear_y(img,tf.random.uniform([],-self.intensity["shearing"],self.intensity["shearing"]),tf.reduce_mean(img))
        img = tfa.image.shear_x(img,tf.random.uniform([],-self.intensity["shearing"],self.intensity["shearing"]),tf.reduce_mean(img))
        return img, label

    def batch_augment(self,img,label):
    # augment for batched image
        img = tf.image.random_brightness(img,self.intensity["brightness"])
        img = tf.image.random_contrast(img,1-self.intensity["contrast"],1+self.intensity["contrast"])
        img = tf.image.random_hue(img,self.intensity["hue"])
        img = tf.image.random_saturation(img,1-self.intensity["saturation"],1+self.intensity["saturation"])   
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tfa.image.rotate(img, tf.random.uniform([],0.,90.), "bilinear", "reflect")
        # gaussian noise
        if tf.random.uniform([]) < 1:
            stddev = tf.random.uniform([],0.,self.intensity["gaussian_std"])
            mean = tf.random.uniform([],-self.intensity["gaussian_mean"],self.intensity["gaussian_mean"])
            img = img + tf.random.normal(shape=tf.shape(img), mean=mean, stddev=stddev, dtype=tf.float32)
        # random cutout
        if tf.random.uniform([]) < 1:
            size = tf.random.uniform([2], int(self.size_h*self.intensity["cutout_min"]), int(self.size_h*self.intensity["cutout_max"]), dtype=tf.int32)
            img = tfa.image.random_cutout(img, size//2*2)
        # resize and padding
        # if tf.random.uniform([]) < 0.1:
        #     # resize on height and padding
        #     if tf.random.uniform([]) < 0.5:
        #         new_h = tf.random.uniform([],self.size_h//10*8,self.size_h,dtype=tf.int32).numpy()
        #         pad_h = tf.tile(tf.reduce_mean(img,axis=1,keepdims=True),[1,self.size_h-new_h,1,1])
        #         img = tf.image.resize(img,[new_h,self.size_w],"bicubic")
        #         # padding on top or on bottom
        #         if tf.random.uniform([]) < 0.5:
        #             img = tf.concat([img,pad_h],axis=1)
        #         else:
        #             img = tf.concat([pad_h,img],axis=1)
        #     else:
        #         new_w = tf.random.uniform([],self.size_w//10*8,self.size_w,dtype=tf.int32).numpy()
        #         pad_w = tf.tile(tf.reduce_mean(img,axis=2,keepdims=True),[1,1,self.size_w-new_w,1])
        #         img = tf.image.resize(img,[self.size_h,new_w],"bicubic")
        #         # padding on left or on right
        #         if tf.random.uniform([]) < 0.5:
        #             img = tf.concat([img,pad_h],axis=2)
        #         else:
        #             img = tf.concat([pad_h,img],axis=2)
        # task dependent augmentation
        # if tf.random.uniform([]) < 0.1:
        #     pick = tf.random.shuffle(tf.constant(["a","b","c","d","e"]))[0]
        #     if tf.math.equal(pick,tf.constant("a")):
        #         size_h = tf.random.uniform([],self.size_h//32,self.size_h//8,dtype=tf.int32).numpy()
        #         size_w = tf.random.uniform([],self.size_w//32,self.size_w//8,dtype=tf.int32).numpy()
        #         anchor_x = tf.random.uniform([],0,self.size_w-size_w,dtype=tf.int32).numpy()
        #         anchor_y = tf.random.uniform([],0,self.size_h-size_h,dtype=tf.int32).numpy()
        #         img = img[:,anchor_y:anchor_y+size_h,anchor_x:anchor_x+size_w,:]
        #         img = tf.image.resize(img,[self.size_h,self.size_w],tf.random.shuffle(self.interpolation)[0])
        #         label = tf.constant([[self.on_value,self.off_value,self.off_value] for _ in range(self.batch)])
        #     elif tf.math.equal(pick,tf.constant("b")):
        #         img = img[:,self.size_h//2+self.size_h//8:,:,:]
        #         img = tf.image.resize(img,[self.size_h,self.size_w],tf.random.shuffle(self.interpolation)[0])
        #         label = tf.constant([[self.on_value,self.off_value,self.off_value] for _ in range(self.batch)])
        #     elif tf.math.equal(pick,tf.constant("c")):
        #         img = img[:,:,:self.size_w//4,:]
        #         img = tf.image.resize(img,[self.size_h,self.size_w],tf.random.shuffle(self.interpolation)[0])
        #         label = tf.constant([[self.on_value,self.off_value,self.off_value] for _ in range(self.batch)])
        #     elif tf.math.equal(pick,tf.constant("d")):
        #         img = img[:,:,self.size_w//4*3:,:]
        #         img = tf.image.resize(img,[self.size_h,self.size_w],tf.random.shuffle(self.interpolation)[0])
        #         label = tf.constant([[self.on_value,self.off_value,self.off_value] for _ in range(self.batch)])
        #     elif tf.math.equal(pick,tf.constant("e")):
        #         size_h = tf.random.uniform([],2,self.size_h//16,dtype=tf.int32).numpy()
        #         size_w = tf.random.uniform([],2,self.size_w//16,dtype=tf.int32).numpy()
        #         img = tf.image.resize(img,[size_h,size_w],tf.random.shuffle(self.interpolation)[0])
        #         img = tf.image.resize(img,[self.size_h,self.size_w],tf.random.shuffle(self.interpolation)[0])
        #         label = tf.constant([[self.on_value,self.off_value,self.off_value] for _ in range(self.batch)])
        return img, label

    def bigger_size(self, h, w):
        self.size_h, self.size_w = h, w
        
    def more_intense(self):
        for augment in self.intensity.keys():
            self.intensity[augment] += self.min_intensity[augment] * self.augment_multiple / self.augment_step
            self.intensity[augment] = min(self.intensity[augment], self.max_intensity[augment])

class TFRecordLoader(Loader):

    def __init__(self,
                train_cfg=None,
                data_cfg=None,
                size_h=128, 
                size_w=128, 
                batch_size=16, 
                val_split=0.1, 
                shuffle=1024, 
                class_name=["class1","class2"],
                wrong_label="undetermined",
                dir_map={"./class1":"class1","./class2":"class2","./frame":"frame"},
                augment_multiple=4,
                augment_step=8,
                min_intensity={
                    "brightness":0.1,
                    "contrast":0.1,
                    "saturation":0.05,
                    "hue":0.05,
                    "shearing":0.05,
                    "gaussian_std":0.1,
                    "gaussian_mean":0.05,
                    "cutout_min":1/20,
                    "cutout_max":1/12,
                    }
                ):
        super(TFRecordLoader, self).__init__(cfg, size_h, size_w, batch_size, val_split, shuffle, class_name, wrong_label, dir_map, augment_multiple, augment_step, min_intensity)
        self.train_dir, self.train_cls, self.val_dir, self.val_cls = [], [], [], []
        self.train, self.val = {}, {}
        self.train_info, self.val_info = dict(zip(self.class_name,[0]*self.num_class)), dict(zip(self.class_name,[0]*self.num_class))
        for map_dir in dir_map.keys():
            map_cls = dir_map[map_dir]
            glob_dir = glob(os.path.join(map_dir,"*.*"))
            # loading .tfrecord
            glob_dir = [i for i in glob_dir if i.endswith(".tfrecord")]
            if len(glob_dir) == 0:
                raise ValueError(f"no tfrecord in {map_dir}")
            random.Random(1).shuffle(glob_dir)
            if map_cls == "frame":
                train_glob_dir = glob_dir
                self.train_dir.extend(train_glob_dir)
                self.train_cls.extend([map_cls]*len(train_glob_dir))
            else:
                val_glob_dir = glob_dir[0:int(val_split*len(glob_dir))]
                self.val_dir.extend(val_glob_dir)
                self.val_cls.extend([map_cls]*len(val_glob_dir))
                self.val_info[map_cls] += len(val_glob_dir)
                print(f"class {map_cls} has {len(val_glob_dir)} validation tfrecords")
                train_glob_dir = glob_dir[int(val_split*len(glob_dir)):]
                self.train_dir.extend(train_glob_dir)
                self.train_cls.extend([map_cls]*len(train_glob_dir))
                self.train_info[map_cls] += len(train_glob_dir)
                print(f"class {map_cls} has {len(train_glob_dir)} training tfrecords")
        if not val_split:
            self.val_cls, self.val_dir, self.val_info = self.train_cls, self.train_dir, self.train_info

    def parse_tfrecord(self, example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
            }
        example = tf.io.parse_single_example(example, feature_description)
        image, label = example['image'], example["label"]
        return image, label

    def get(self,split):
        if split == "train":
            ds_dir = tf.data.Dataset.from_tensor_slices(self.train_dir)
            ds_cls = tf.data.Dataset.from_tensor_slices(self.train_cls)
            ds = tf.data.Dataset.zip((ds_dir,ds_cls)).cache()
            ds = ds.shuffle(self.shuffle,reshuffle_each_iteration=True)
            ds = ds.map(lambda x,y: tf.py_function(self.train_preprocess,[x,y],[tf.float32,tf.float32]),num_parallel_calls=self.AUTOTUNE)
            ds = ds.map(self.single_augment,num_parallel_calls=self.AUTOTUNE)
            ds = ds.batch(self.batch_size,drop_remainder=True)
            ds = ds.map(lambda x,y: tf.py_function(self.batch_augment,[x,y,],[tf.float32,tf.float32]),num_parallel_calls=self.AUTOTUNE)
            ds = ds.prefetch(self.AUTOTUNE)
        elif split == "val":
            ds_dir = tf.data.Dataset.from_tensor_slices(self.val_dir)
            ds_cls = tf.data.Dataset.from_tensor_slices(self.val_cls)
            ds = tf.data.Dataset.zip((ds_dir,ds_cls)).cache()
            ds = ds.map(self.val_preprocess,num_parallel_calls=self.AUTOTUNE)
            ds = ds.batch(self.batch_size,drop_remainder=True)
            ds = ds.prefetch(self.AUTOTUNE)
        elif split == "info":
            # return number of total train images, total val images, train images per class, val images per class
            ds = [len(self.train_cls),len(self.val_cls),self.train_info,self.val_info]
        else:
            raise ValueError("wrong split parameters")
        return ds

class ImageLoader(Loader):

    def __init__(self,
                train_cfg=None,
                data_cfg=None,
                size_h=128, 
                size_w=128, 
                batch_size=16, 
                val_split=0.1, 
                shuffle=1024, 
                class_name=["class1","class2"],
                wrong_label="undetermined",
                dir_map={"./class1":"class1","./class2":"class2"},
                augment_multiple=4,
                augment_step=8,
                min_intensity={
                    "brightness":0.05,
                    "contrast":0.05,
                    "saturation":0.025,
                    "hue":0.025,
                    "shearing":0.025,
                    "gaussian_std":0.05,
                    "gaussian_mean":0.0,
                    "cutout_min":1/24,
                    "cutout_max":1/20,
                    }
                ):
        super(ImageLoader, self).__init__(train_cfg, data_cfg, size_h, size_w, batch_size, val_split, shuffle, class_name, wrong_label, dir_map, augment_multiple, augment_step, min_intensity)
        self.train_dir, self.train_cls, self.val_dir, self.val_cls = [], [], [], []
        self.train, self.val = {}, {}
        self.train_info, self.val_info = dict(zip(self.class_name,[0]*self.num_class)), dict(zip(self.class_name,[0]*self.num_class))
        for map_dir in dir_map.keys():
            map_cls = dir_map[map_dir]
            glob_dir = glob(os.path.join(map_dir,"*.*"))
            # loading .jpg, .jepg and .png
            glob_dir = [i for i in glob_dir if i.endswith(".jpg") or i.endswith(".jpeg") or i.endswith(".png")]    
            if len(glob_dir) == 0:
                raise ValueError(f"no images in {map_dir}")
            random.Random(1).shuffle(glob_dir)
            if map_cls == "frame":
                train_glob_dir = glob_dir
                self.train_dir.extend(train_glob_dir)
                self.train_cls.extend([map_cls]*len(train_glob_dir))
            else:
                val_glob_dir = glob_dir[0:int(val_split*len(glob_dir))]
                self.val_dir.extend(val_glob_dir)
                self.val_cls.extend([map_cls]*len(val_glob_dir))
                self.val_info[map_cls] += len(val_glob_dir)
                #print(f"class {map_cls} has {len(val_glob_dir)} validation images")
                train_glob_dir = glob_dir[int(val_split*len(glob_dir)):]
                self.train_dir.extend(train_glob_dir)
                self.train_cls.extend([map_cls]*len(train_glob_dir))
                self.train_info[map_cls] += len(train_glob_dir)
                #print(f"class {map_cls} has {len(train_glob_dir)} training images")
        if not val_split:
            self.val_cls, self.val_dir, self.val_info = self.train_cls, self.train_dir, self.train_info
        print(self.train_info)
        print(self.val_info)
        self.train = dict(zip(self.train_dir,self.train_cls))
        self.train_dir = list(self.train.keys())
        random.shuffle(self.train_dir)
        self.train_cls = [self.train.get(i) for i in self.train_dir]
        print(f"total {len(self.train_cls)} train images and {len(self.val_cls)} val images")

    def train_read(self, img, label):
        # img read and labeling
        img = tf.io.read_file(img)
        img = tf.io.decode_image(img, channels=3, dtype=tf.float32, expand_animations=False)
        # label smoothing for blurry/small images
        if min(tf.shape(img)[:2]) < self.size_w//8 and label != "background":
            img = tf.image.resize(img,[self.size_h,self.size_w],"bicubic")
            on_value = self.on_value-0.3
            off_value = (1. - on_value)/(self.num_class-1)
            label = tf.argmax(label == self.class_name)
            label = tf.one_hot(label, len(self.class_name), on_value, off_value)
        elif min(tf.shape(img)[:2]) < self.size_w//4 and label != "background":
            img = tf.image.resize(img,[self.size_h,self.size_w],"bicubic")
            on_value = self.on_value-0.15
            off_value = (1. - on_value)/(self.num_class-1)
            label = tf.argmax(label == self.class_name)
            label = tf.one_hot(label, len(self.class_name), on_value, off_value)   
        # random crop for label "frame"              
        elif tf.math.equal(label,"frame"):
            size_h = tf.random.uniform([],tf.shape(img)[0]//32,tf.shape(img)[0]//4,dtype=tf.int32).numpy()
            size_w = tf.random.uniform([],tf.shape(img)[1]//32,tf.shape(img)[1]//4,dtype=tf.int32).numpy()
            anchor_x = tf.random.uniform([],0,tf.shape(img)[1]-size_w,dtype=tf.int32).numpy()
            anchor_y = tf.random.uniform([],0,tf.shape(img)[0]-size_h,dtype=tf.int32).numpy()
            img = img[anchor_y:anchor_y+size_h,anchor_x:anchor_x+size_w,:]
            img = tf.image.resize(img,[self.size_h,self.size_w],"bicubic")
            label = tf.argmax(self.wrong_label == self.class_name)
            label = tf.one_hot(label, len(self.class_name), self.on_value, self.off_value)
        # normal sized training images
        else:
            img = tf.image.resize(img,[self.size_h,self.size_w],"bicubic")
            label = tf.argmax(label == self.class_name)
            label = tf.one_hot(label, len(self.class_name), self.on_value, self.off_value)
        return img, label

    def val_read(self, img, label):
        img = tf.io.read_file(img)
        img = tf.io.decode_image(img,channels=3,dtype=tf.float32,expand_animations=False)
        img = tf.image.resize(img,[self.size_h,self.size_w],"bicubic")
        label = tf.argmax(label == self.class_name)
        label = tf.one_hot(label, len(self.class_name), self.on_value, self.off_value)
        return img, label

    def get(self, split):
        if split == "train":
            ds_dir = tf.data.Dataset.from_tensor_slices(self.train_dir)
            ds_cls = tf.data.Dataset.from_tensor_slices(self.train_cls)
            ds = tf.data.Dataset.zip((ds_dir,ds_cls))
            ds = ds.shuffle(self.shuffle,reshuffle_each_iteration=True)
            ds = ds.map(lambda x,y: tf.py_function(self.train_read,[x,y],[tf.float32,tf.float32]),num_parallel_calls=self.AUTOTUNE)
            ds = ds.map(self.single_augment,num_parallel_calls=self.AUTOTUNE)
            ds = ds.batch(self.batch_size,drop_remainder=True)
            ds = ds.map(lambda x,y: tf.py_function(self.batch_augment,[x,y,],[tf.float32,tf.float32]),num_parallel_calls=self.AUTOTUNE)
            ds = ds.prefetch(self.AUTOTUNE)
        elif split == "val":
            ds_dir = tf.data.Dataset.from_tensor_slices(self.val_dir)
            ds_cls = tf.data.Dataset.from_tensor_slices(self.val_cls)
            ds = tf.data.Dataset.zip((ds_dir,ds_cls))
            ds = ds.map(self.val_read,num_parallel_calls=self.AUTOTUNE)
            ds = ds.batch(self.batch_size,drop_remainder=True)
            ds = ds.prefetch(self.AUTOTUNE)
        elif split == "info":
            # return number of total train images, total val images, train images per class, val images per class
            ds = [len(self.train_cls),len(self.val_cls),self.train_info,self.val_info]
        else:
            raise ValueError("wrong split parameters")
        return ds