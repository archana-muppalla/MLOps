import os
import re
import string

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

class DataPreprocessing:
    def __init__(self,batch_size=16,epochs=30) -> None:
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.IMAGE_SIZE = (299,299)
        self.AUTOTUNE = tf.data.AUTOTUNE

    def get_text_features(self,caption_list,vocab_size=10000,seq_len=20):
        self.vectorization = TextVectorization(
                            max_tokens=vocab_size,
                            output_mode="int",
                            output_sequence_length=seq_len,
                            standardize=self.custom_standardization,
                        )
        self.vectorization.adapt(caption_list)
        return self.vectorization

    def custom_standardization(self,input_string):
        strip_chars = string.punctuation
        strip_chars = strip_chars.replace("<", "")
        strip_chars = strip_chars.replace(">", "")
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
    
    def decode_and_resize(self,img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img


    def process_input(self,img_path, captions):
        return self.decode_and_resize(img_path), self.vectorization(captions)


    def make_dataset(self,images, captions):

        dataset = tf.data.Dataset.from_tensor_slices((images, captions))
        dataset = dataset.shuffle(len(images))
        dataset = dataset.map(self.process_input, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(self.AUTOTUNE)
        return dataset


