import tensorflow as tf
import tensorflow_addons as tfa

class Augmenter:
    
    def __init__(self, task, min_int, max_int):
        self.task = task
        self.min_int = min_int
        self.max_int = max_int
        self.pos_neg = [1., -1.]

    def __random_int(self, min_value, max_value):
        return tf.random.uniform([], min_value, max_value, dtype=tf.int32)

    def __random_float(self, min_value, max_value):
        return tf.random.uniform([], min_value, max_value, dtype=tf.float32)

    def __random_sampling(self, array, num_samples):
        return array[tf.random.categorical([[1./len(array)]*len(array)], num_samples)]

    def __blend(image1, image2, factor):
        """
        Factor can be above 0.0.  A value of 0.0 means only image1 is used.
        A value of 1.0 means only image2 is used.  A value between 0.0 and
        1.0 means we linearly interpolate the pixel values between the two
        images.
        """
        # Do addition in float.
        blended = image1 + factor * (image2 - image1)
        return tf.clip_by_value(blended, 0.0, 1.0)

    def _random_brightness(self):
        """Equivalent of PIL Brightness."""
        min_int = self.min_int["brightness"]
        max_int = self.max_int["brightness"]
        degenerate = tf.zeros_like(image)
        intensity = self.__random_sampling(self.pos_neg, 1) * self.__random_float(min_int, max_int)
        self.image = self.__blend(degenerate, self.image, intensity)

    def contrast(self):
        """Equivalent of PIL Contrast."""
        degenerate = tf.image.rgb_to_grayscale(image)
        # Cast before calling tf.histogram.
        degenerate = tf.cast(degenerate, tf.int32)

        # Compute the grayscale histogram, then compute the mean pixel value,
        # and create a constant image size of that value.  Use that as the
        # blending degenerate target of the original image.
        hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
        return blend(degenerate, image, factor)    

    def run(self, image):
        if self.task == "classification":
            self.run = self._classification_run
        elif self.task == "super-resolution":
            self.run = self._super_resolution_run

    def _classification_run(self, image):
        self.image = image
        ### uint operations

        ### cast
        self.image = tf.cast(self.image, tf.float32) / 255.
        ### float operations
