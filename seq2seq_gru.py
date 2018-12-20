# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:34:20 2018

@author: Hew
"""

from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'dan-eng/dan.txt'

# Vectorise the data
input_texts = []
target_texts = []
input_words = []
input_words_freq = {}
target_words = []
target_words_freq = {}
with open(data_path,'r',encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples,len(lines)-1)]:
    input_text,target_text = line.lower().split('\t')
    target_text = '<START> '+target_text+' <END>'
    input_text = re.sub('([.,!?()])', r' \1 ', input_text)
    input_text = re.sub('\s{2,}', ' ', input_text)
    target_text = re.sub('([.,!?()])', r' \1 ', target_text)
    target_text = re.sub('\s{2,}', ' ', target_text)
    input_text = input_text.strip()
    target_text = target_text.strip()
    input_texts.append(input_text)
    target_texts.append(target_text)
    for word in input_text.split(' '):
        if not word in input_words_freq:
                input_words_freq[word]=1
        else:
            input_words_freq[word]+=1     
    for word in target_text.split(' '):
        if not word in target_words_freq:
                target_words_freq[word]=1
        else:
            target_words_freq[word]+=1 
        
            
input_replace = set()
target_replace = set()
for key,value in input_words_freq.copy().items():
    if value<1:
        del input_words_freq[key]
        input_replace.add(key)
        
for key,value in target_words_freq.copy().items():
    if value<1:
        del target_words_freq[key]
        target_replace.add(key)
        

# get lists of words including pad and unknown tokens
input_words = ['<PAD>']+['<UNKNOWN>']+sorted(list(input_words_freq.keys()))
target_words = ['<PAD>']+['<UNKNOWN>']+sorted(list(target_words_freq.keys()))
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
max_encoder_seq_length = max([len(txt.split(' ')) for txt in input_texts])
max_decoder_seq_length = max([len(txt.split(' ')) for txt in target_texts])            

#For using embedded vectors for each word
#This section turns texts into integer vectors with padding to max sentence length

input_token_index = dict(
        [(word,i) for i,word in enumerate(input_words)])
target_token_index = dict(
        [(word,i) for i,word in enumerate(target_words)])

encoder_input_data = np.zeros(
        (len(input_texts),max_encoder_seq_length),
        dtype='float64')
decoder_input_data = np.zeros(
        (len(input_texts),max_decoder_seq_length),
        dtype='float64')
decoder_target_data = np.zeros(
        (len(input_texts),max_decoder_seq_length,num_decoder_tokens),
        dtype='float32')

for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):
    for t,word in enumerate(input_text.split(' ')):
        if word in input_replace:
            word = '<UNKNOWN>'
        encoder_input_data[i,t]=input_token_index[word]
    for t,word in enumerate(target_text.split(' ')):
        if word in target_replace:
            word = '<UNKNOWN>'
        decoder_input_data[i,t]=target_token_index[word]
        if t>0:
            decoder_target_data[i,t-1,target_token_index[word]]=1
            
# Pre-trained embeddings
embeddings_index = dict()
f = open('/glove6b/glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    if word in input_words:
        coefs = np.asarray(values[1:], dtype='float64')
        embeddings_index[word] = coefs
        if len(embeddings_index)==num_encoder_tokens:
            break
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((num_encoder_tokens, 100))
for word, i in input_token_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
            
# GRU Encoder
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, 100, weights=[embedding_matrix], trainable=True)(encoder_inputs)
encoder_gru = GRU(latent_dim,return_state=True)
encoder_outputs, encoder_state = encoder_gru(x)

# GRU Decoder
decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_gru = GRU(latent_dim,return_sequences=True,return_state=True)
decoder_outputs, decoder_state = decoder_gru(x,
                                             initial_state=encoder_state)
decoder_dense = Dense(num_decoder_tokens,activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Seq2seq model
model = Model([encoder_inputs,decoder_inputs],decoder_outputs)

# Train
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
model.fit([encoder_input_data,decoder_input_data],decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          verbose=1)

# Save
model.save('s2s.h5')

# Load
#model = load_model('fra-eng/s2s.h5')

encoder_model = Model(encoder_inputs,encoder_state)

decoder_state_input = Input(shape=(latent_dim,))
decoder_outputs,decoder_state = decoder_gru(x,
                                            initial_state=decoder_state_input)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs]+[decoder_state_input],
                      [decoder_outputs]+[decoder_state])

reverse_input_word_index = dict(
        (i,word) for word,i in input_token_index.items())
reverse_target_word_index = dict(
        (i,word) for word,i in target_token_index.items())


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    
    decoder_input_seq = [target_token_index['<START>']]
    
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens,states_value = decoder_model.predict([decoder_input_seq]+[states_value])
        
        sample_token_index = np.argmax(output_tokens[0,-1,:])
        
        sampled_word = reverse_target_word_index[sample_token_index]
            
        if (sampled_word == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        elif (sampled_word!='<PAD>'):
            decoded_sentence.append(' ')
            decoded_sentence.append(sampled_word)
        
        decoder_input_seq=[target_token_index[sampled_word]]
    
    decoded_string=''
    for word in decoded_sentence:
        decoded_string+=word
        
    return decoded_string



if __name__=='__main__':
    for seq_index in range(100):
        input_seq = encoder_input_data[seq_index:seq_index+1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)
        
        
        






            
        



            
    