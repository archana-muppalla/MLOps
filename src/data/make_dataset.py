import tensorflow as tf
import os
import numpy as np

class Dataset:
    def __init__(self,path="Flicker8k_Dataset", img_size=(299,299),vocab_size=10000,seq_len=25,
                embed_dim=512,ff_dim=512,batch_size=64,epochs=30,) -> None:
        self.IMAGES_PATH = path

        self.IMAGE_SIZE = img_size

        self.VOCAB_SIZE = vocab_size

        self.SEQ_LENGTH = seq_len

        self.EMBED_DIM = embed_dim

        self.FF_DIM = ff_dim

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.AUTOTUNE = tf.data.AUTOTUNE

    def buildCaptionsMap(self,filename):
        """
        
        """
        with open(filename) as caption_file:
            caption_data = caption_file.readlines()
            caption_map = {}
            caption_list = []
            images_to_skip = set()

            for line in caption_data:
                line = line.rstrip("\n")

                if len(line.split("\t"))!=2:
                    continue

                img_name, caption = line.split("\t")

                img_name = img_name.split("#")[0]
                img_name = os.path.join(self.IMAGES_PATH, img_name.strip())

                if img_name.endswith("jpg") and img_name not in images_to_skip:

                    caption = "<start> " + caption.strip() + " <end>"
                    caption_list.append(caption)

                    if img_name in caption_map:
                        caption_map[img_name].append(caption)
                    else:
                        caption_map[img_name] = [caption]

            for img_name in images_to_skip:
                if img_name in caption_map:
                    del caption_map[img_name]

            return caption_map, caption_list
    
    def buildTrainAndValidationSet(self, filename, caption_map):
        with open(filename) as train_and_val_file:
            train_and_val_images = train_and_val_file.readlines()

            for i in range(len(train_and_val_images)):
                line = train_and_val_images[i]
                line = line.rstrip("\n")
                line = os.path.join(self.IMAGES_PATH, line.strip())
                train_and_val_images[i] = line

            np.random.shuffle(train_and_val_images)
            train_size = int(0.8 * len(train_and_val_images))
            
            train_data = {
                img_name: caption_map[img_name] for img_name in train_and_val_images[:train_size]
            }

            validation_data = {
                img_name: caption_map[img_name] for img_name in train_and_val_images[train_size:]
            }

        return train_data, validation_data
    
    def getTestData(self,filename,caption_map):
        with open(filename) as test_file:
            test_images = test_file.readlines()

            for i in range(len(test_images)):
                line = test_images[i]
                line = line.rstrip("\n")
                line = os.path.join(self.IMAGES_PATH, line.strip())
                test_images[i] = line

            test_data = {
                img_name: caption_map[img_name] for img_name in test_images
            }

        return test_data