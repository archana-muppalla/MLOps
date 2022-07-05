import sys
from urllib.parse import urlparse

sys.path.append('/home/archana/projects/MLOps/src/')
from data.make_dataset import Dataset
from tensorflow.keras.applications import efficientnet
from tensorflow.keras import layers
from tensorflow import keras
from features.build_features import DataPreprocessing
from helper import TransformerEncoder, PositionalEmbedding, TransformerDecoder, BuildModel
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from nltk.translate.bleu_score import corpus_bleu
import mlflow
from keras.models import model_from_json
from keras.models import load_model

mlflow.set_tracking_uri("https://dagshub.com/archana/MLOps.mlflow")
tracking_uri = mlflow.get_tracking_uri()

class ImageCaptioningModel:
    def __init__(self,img_size=(299,299),vocab_size=10000,seq_len=20,
                embed_dim=512,ff_dim=512,batch_size=64,epochs=30,) -> None:
        self.IMAGE_SIZE = img_size
        self.VOCAB_SIZE = vocab_size
        self.SEQ_LENGTH = seq_len
        self.EMBED_DIM = embed_dim
        self.model_file = "tmp/model"
        self.FF_DIM = ff_dim
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.data_path = "/home/archana/projects/MLOps/src/data/raw/"
        self.images_path = self.data_path+"Images"
        self.tokens_path = self.data_path+"Flickr8k.token.txt"
        self.train_images_path = self.data_path+"Flickr_8k.trainImages.txt"
        self.test_images_path = self.data_path+"Flickr_8k.testImages.txt"
        self.dataset = Dataset(path=self.images_path)
        self.caption_map, self.caption_list = self.get_tokens_data()
        self.train_data, self.validation_data = self.get_train_val_data()
        self.data_preprocess = DataPreprocessing()
        self.image_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.25),
            ]
        )
        print("Image path")
        print(list(self.train_data.keys())[0])
        print()
        self.test_data = self.dataset.getTestData(self.test_images_path,self.caption_map)

    def get_tokens_data(self):
        return self.dataset.buildCaptionsMap(self.tokens_path)
    
    def get_train_val_data(self):
        return self.dataset.buildTrainAndValidationSet(self.train_images_path, self.caption_map)
    
    

    def get_cnn_model(self):
        base_model = efficientnet.EfficientNetB0(
            input_shape=(*self.IMAGE_SIZE, 3), include_top=False, weights="imagenet",
        )
        
        base_model.trainable = False
        base_model_out = base_model.output
        base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
        cnn_model = keras.models.Model(base_model.input, base_model_out)
        return cnn_model
    


    def calculate_bleu(self):

        max_decoded_sentence_length = self.SEQ_LENGTH-1

        image_keys = list(self.test_data.keys())
        captions_array = list(self.test_data.values())

        predicted_captions = []

        actual_captions = []
        for captions in captions_array:
            sentence_captions = []
            for caption in captions:
                sentence_captions.append(caption)
                actual_captions.append(sentence_captions)
            

        for image_key in image_keys:

            image = self.data_preprocess.decode_and_resize(image_key)
            
            img = tf.expand_dims(image, 0)
            img = self.caption_model.cnn_model(img)

            encoded_img = self.caption_model.encoder(img, training=False)

            decoded_caption = "<start> "
            for i in range(max_decoded_sentence_length):
                tokenized_caption = self.vectorizer([decoded_caption])[:, :-1]
                mask = tf.math.not_equal(tokenized_caption, 0)
                predictions = self.caption_model.decoder(
                    tokenized_caption, encoded_img, training=False, mask=mask
                )
                sampled_token_index = np.argmax(predictions[0, i, :])
                sampled_token = self.index_lookup[sampled_token_index]
                if sampled_token == " <end>":
                    break
                decoded_caption += " " + sampled_token

            decoded_caption = decoded_caption.replace("<start> ", "")
            decoded_caption = decoded_caption.replace(" <end>", "").strip()
            predicted_captions.append(decoded_caption)

        bleu_score = corpus_bleu(actual_captions, predicted_captions)

        return bleu_score
    
    def save_model(self):
        self.caption_model.save_weights("model_num.h5")

    def train(self):
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        model_dir = os.listdir(self.model_file)
        self.cnn_model = self.get_cnn_model()
        encoder = TransformerEncoder(embed_dim=self.EMBED_DIM, dense_dim=self.FF_DIM, num_heads=1)
        decoder = TransformerDecoder(embed_dim=self.EMBED_DIM, ff_dim=self.FF_DIM, num_heads=2,seq_len=self.SEQ_LENGTH,
                                        vocab_size = self.VOCAB_SIZE)
        self.caption_model = BuildModel(
                cnn_model=self.cnn_model, encoder=encoder, decoder=decoder, image_aug=self.image_augmentation
            )
        cross_entropy = keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, reduction="none"
            )
        self.caption_model.compile(optimizer=keras.optimizers.Adam(), loss=cross_entropy)
        history = self.caption_model.fit(
            self.train_dataset,
            epochs=1,
            validation_data=self.validation_dataset,
            callbacks=[cp_callback]
        )
        train_loss=history.history['loss'][-1]
        train_acc=history.history['sparse_categorical_accuracy'][-1]
        val_loss=history.history['val_loss'][-1]
        val_acc=history.history['val_sparse_categorical_accuracy'][-1]

        #self.save_model()
        
        #self.caption_model.save(self.model)

        #tf.keras.models.save_model(self.caption_model, self.model_file)

    def model(self):
        self.vectorizer = self.data_preprocess.get_text_features(self.caption_list)
        print(self.vectorizer("black dog running in the grass"))
        self.train_dataset = self.data_preprocess.make_dataset(list(self.train_data.keys()), 
        list(self.train_data.values()))
        self.validation_dataset = self.data_preprocess.make_dataset(list(self.validation_data.keys()), 
        list(self.validation_data.values()))
        print(self.train_dataset)
        self.train()
        #tf.keras.models.save_model(self.caption_model, 'model.h5')
        self.get_testing_accuracy()

    def get_testing_accuracy(self):
        self.vocab = self.vectorizer.get_vocabulary()
        self.index_lookup = dict(zip(range(len(self.vocab)), self.vocab))
        bleu_score = self.calculate_bleu()
        print(bleu_score)
        return bleu_score
    
if __name__ == "__main__":
    img = ImageCaptioningModel()
    with mlflow.start_run():
        img.model()
        bleu_score = img.get_testing_accuracy()
        mlflow.log_metric("bleu_score", bleu_score)