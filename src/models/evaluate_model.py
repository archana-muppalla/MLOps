import os 
import sys

sys.path.append('/home/archana/projects/MLOps/src/')
from tensorflow import keras
from helper import TransformerEncoder, TransformerDecoder, BuildModel
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from model import ImageCaptioningModel
from features.build_features import DataPreprocessing


class TestModel:
    def __init__(self,vocab_size=10000,seq_len=20,embed_dim=512,ff_dim=512):
        self.VOCAB_SIZE = vocab_size
        self.SEQ_LENGTH = seq_len
        self.EMBED_DIM = embed_dim
        self.FF_DIM = ff_dim
        self.image_cap = ImageCaptioningModel()
        self.image_augmentation = self.image_cap.image_augmentation
        self.data_preprocess = DataPreprocessing()
        #self.IMAGES_PATH = path
        self.data_path = "/home/archana/projects/MLOps/src/data/raw/"
        self.images_path = self.data_path+"Images"
        self.caption_map, self.caption_list = self.image_cap.get_tokens_data()

    # def get_tokens_data(self):
    #     return self.dataset.buildCaptionsMap(self.tokens_path)

    def predict(self,img_path):
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # # Create a callback that saves the model's weights
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                                 save_weights_only=True,
        #                                                 verbose=1)

        # model_dir = os.listdir(self.model_file)
        self.cnn_model = self.image_cap.get_cnn_model()
        if os.path.exists(checkpoint_dir):
            encoder = TransformerEncoder(embed_dim=self.EMBED_DIM, dense_dim=self.FF_DIM, num_heads=1)
            decoder = TransformerDecoder(embed_dim=self.EMBED_DIM, ff_dim=self.FF_DIM, num_heads=2,
                                         seq_len=self.SEQ_LENGTH,
                                        vocab_size = self.VOCAB_SIZE)
            self.caption_model = BuildModel(
                cnn_model=self.cnn_model, encoder=encoder, decoder=decoder, image_aug=self.image_augmentation
            )
            cross_entropy = keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, reduction="none"
            )
            self.caption_model.compile(optimizer=keras.optimizers.Adam(), loss=cross_entropy)
            self.caption_model.built = True
            img_path = img_path.rstrip("\n")
            img_path = os.path.join(self.images_path, img_path.strip())
            #sample_img = self.data_preprocess.decode_and_resize(img_path)
            #encoded_img = tf.expand_dims(sample_img, 0)
            #self.caption_model(encoded_img)
            self.caption_model.load_weights(checkpoint_path)
            self.vectorizer = self.data_preprocess.get_text_features(self.caption_list)
            self.vocab = self.vectorizer.get_vocabulary()
            self.index_lookup = dict(zip(range(len(self.vocab)), self.vocab))
            self.generate_caption(img_path)
        else:
            print("Model does not exist")

    def generate_caption(self,img):
        max_decoded_sentence_length = self.SEQ_LENGTH -2
        sample_img = self.data_preprocess.decode_and_resize(img)
        encoded_img = tf.expand_dims(sample_img, 0)
        decoded_caption = "<start> "
        for i in range(max_decoded_sentence_length):
            tokenized_caption = self.vectorizer([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = self.caption_model([encoded_img,tokenized_caption])
            print(predictions)
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = self.index_lookup[sampled_token_index]
            if sampled_token == " <end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        print("Predicted Caption: ", decoded_caption)

if __name__ == "__main__":
    model = TestModel()
    img_path = input("Enter Image Path")
    #img.predict("3385593926_d3e9c21170.jpg")
    model.predict(img_path)