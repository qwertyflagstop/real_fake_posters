from keras import objectives, backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, CuDNNLSTM, CuDNNGRU, RepeatVector, TimeDistributed, Dropout
from keras.models import Model, Sequential
from keras.layers import Conv2D, Lambda, Input, ZeroPadding2D, add, LeakyReLU, BatchNormalization, MaxPooling2D, Layer, concatenate, Activation, SeparableConv2D, GlobalAveragePooling2D
from keras.models import Model, model_from_json
import keras.backend as K
from keras.utils.generic_utils import Progbar
from keras import layers
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import plot_model
from batchmaker import BatchMaker, BatchGenerator
from logger import Logger
from h5py import File
import os
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


import numpy as np
from keras.applications import ResNet50
import string
import json
from math import floor

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def xception_base_network(input, padding=False):
    weights_path = get_file('xception_weights.h5',TF_WEIGHTS_PATH_NO_TOP,cache_subdir='models')
    #x = Lambda(lambda x: x*2-1)(img_input)
    blocks = []
    x = Conv2D(32, (3, 3), padding='same', use_bias=False, name='block1_conv1')(input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    blocks.insert(0, x)
    x = Conv2D(64, (3, 3),  strides=(2, 2), use_bias=False, name='block1_conv2', padding='same')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    blocks.insert(0,x)
    residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)


    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    blocks.insert(0,x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    blocks.insert(0,x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    blocks.insert(0,x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)
        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    blocks.insert(0, x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalAveragePooling2D()(x)

    model = Model(input, x, name='xception')

    model.load_weights(weights_path)
    return model, x, blocks

class TextBatchMaker(BatchMaker):

    def __init__(self, name):
        super(TextBatchMaker, self).__init__()
        self.name = name
        path = '/media/qwertyflagstop/data/{}.h5'.format(self.name)
        self.file = File(path, mode='r')
        self.epoch_length = len(self.file.keys())
        # self.chars_to_indexes = {}
        # self.indexes_to_chars = {}
        # self.chars_to_indexes[0] = 'END'
        # for i,c in enumerate(string.printable.lower()):
        #     self.chars_to_indexes[c]=i+1
        #     self.indexes_to_chars[i+1]=c
        self.vocab_size = 50
        self.max_len = 100
        texts = []
        for k in self.file.keys():
            movie_facts = np.array(self.file['{}/{}'.format(k, 'plot')]).tostring()
            movie_facts = json.loads(movie_facts)
            if not movie_facts['Language'] == 'English':
                continue
            l = movie_facts['Plot']
            l = l.lower()
            if len(l) < self.max_len*0.8:
                continue
            texts.append(l)
        print('found {} plots'.format(len(texts)))
        self.tokenizer = Tokenizer(self.vocab_size, char_level=True)
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index  # the dict values start from 1 so this is fine with zeropadding
        self.index2word = {v: k for k, v in self.word_index.items()}
        print('Found %s unique tokens' % len(self.word_index))
        sequences = self.tokenizer.texts_to_sequences(texts)
        with open('counts.json','w') as fp:
            json.dump(self.index2word,fp, sort_keys=True, indent=4, separators=(',', ': '))
        self.sentances = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        print('Shape of data tensor:', self.sentances.shape)

    def get_embeding_layer(self, embeding_dim):
        embeddings_index = {}
        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        embedding_matrix = np.zeros((len(self.word_index) + 1, embeding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_layer = Embedding(len(self.word_index) + 1, embeding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_len,
                                    trainable=False)

        return embedding_layer


    def get_batch(self, offset, batch_size):
        j = np.random.randint(0,self.sentances.shape[0]-batch_size-1)
        X = self.sentances[j:j+batch_size]
        y = []
        for i in np.arange(0,X.shape[0]):
            Y = to_categorical(X[i], self.vocab_size)
            y.append(Y)
        return X,np.array(y)

    def decode_ints(self, ints):
        sentance = ''
        for index in ints:
            if index==0: #0 means we done
                break
            sentance +=' {}'.format(self.index2word[index].encode('ascii','ignore'))
        return sentance

class PosterNet(object):

    def __init__(self, name, batch_maker):
        super(PosterNet, self).__init__()
        self.name = name
        self.batch_maker = batch_maker
        self.model = self.make_model()
        self.logger = Logger()
        self.batches = 0

    def make_model(self):
        #input = Input((None, None, 3))
        #encoded_image = self.cnn_encoder(input)
        #input = Input((1024,))
        inp = Input(shape=(self.batch_maker.max_len,))
        x = Embedding(self.batch_maker.vocab_size, 128, input_length=self.batch_maker.max_len)(inp)
        # x = CuDNNLSTM(256, return_sequences=True)(x)
        # x = CuDNNLSTM(256, return_sequences=True)(x)
        # x = CuDNNLSTM(256, return_sequences=True)(x)
        # x = TimeDistributed(Dense(256, activation='relu'))(x)
        # x = TimeDistributed(Dense(self.batch_maker.vocab_size))(x)
        # x = TimeDistributed(Activation('softmax'))(x)
        vae_loss, encoded = self.build_encoder(x, latent_rep_size=128, max_length=self.batch_maker.max_len)
        decoded = self.build_decoder(encoded, self.batch_maker.max_len, self.batch_maker.vocab_size)
        model = Model(inputs=[inp], outputs=[decoded])
        plot_model(model, '{}.png'.format(self.name), show_shapes=True, show_layer_names=True)
        print(model.summary(line_length=150))
        model.compile(optimizer='Adam', loss=[vae_loss], metrics=['accuracy'])
        return model

    def cnn_encoder(self, input):
        m, end_of_net, blocks = xception_base_network(input)
        return blocks[0]

    def build_encoder(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        h = CuDNNLSTM(256, return_sequences=True, name='lstm_1')(x)
        h = CuDNNLSTM(256, return_sequences=False, name='lstm_2')(h)
        h = Dense(256, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def build_decoder(self, input, max_output_length, vocab_size):
        repeated_context = RepeatVector(max_output_length)(input)
        h = CuDNNLSTM(256, return_sequences=True, name='dec_lstm_1')(repeated_context)
        h = CuDNNLSTM(256, return_sequences=True, name='dec_lstm_2')(h)

        decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)

        return decoded

    def train(self):
        min_loss = 99999
        while True:
            X, Y = self.batch_maker.get_batch(0, 512)
            loss = self.model.train_on_batch(X, Y)
            self.logger.log_scalar('Loss', value=loss[0], step=self.batches)
            self.logger.log_scalar('Acc', value=loss[1], step=self.batches)
            self.batches += 1
            if self.batches%100==0:
                min_loss = loss[0]
                self.model.save_weights('{}.h5'.format(self.name))
                X, Y = self.batch_maker.get_batch(0, 10)
                hots = self.model.predict(X)
                for i in np.arange(0,10):
                    print('\n')
                    indexs = np.argmax(hots[i], axis=-1)
                    #sents = self.batch_maker.decode_ints(X[i])
                    print(X[i])
                    #sents = self.batch_maker.decode_ints(indexs)
                    print(indexs)





if __name__ == '__main__':
    bm = TextBatchMaker('movies')
    pn = PosterNet('PNet',bm)
    pn.train()