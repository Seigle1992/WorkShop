# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:16:27 2021

@author: 夏欣雨
"""

import random
import os

import numpy as np
import tensorflow as tf
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def preprocess_file(file):
    files_content = ''
    with open(file, 'r',encoding='utf-8') as f:
        for line in f:
            x = line.strip() + "]"
            x = x.split(":")[1]
            if len(x) <= 5 :
                continue
            if x[5] == '，':
                files_content += x

    words = sorted(list(files_content))
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1
    #去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*wordPairs)
    words += (" ",)
    # word到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, len(words) - 1)
    return word2numF, num2word, words, files_content

def data_generator():
    i = 0
    while 1:
        y_train = np.zeros(shape=(1, len(words)),dtype='int')
        y_train[0, word2numF(y[i])] = 1

        X_train = np.zeros(shape=(1, 6, len(words)),dtype='int')

        for t, char in enumerate(X[i]):
            X_train[0, t, word2numF(char)] = 1

        yield X_train, y_train
        i += 1

def create_model():
    input = Input(shape = (6,len(words)))
    Lstm  = LSTM(256, return_sequences = True)(input)
    dropout = Dropout(rate=0.5)(Lstm)
    Lstm  = LSTM(128, return_sequences = False)(dropout)
    dropout = Dropout(rate=0.5)(Lstm)
    dense = Dense(len(words), activation='softmax')(dropout)
    
    model = Model(input, dense)  
    model.summary()
    
    return model

def cang_tou(text):
    max_len = 6
    Mypoem =  np.zeros(shape=(4,5),dtype='str')
    index = random.randint(0, poems_num)
    #sentence = sentence[-max_len:]
    for i in range(4):
        if i != 0:
            sentence = (sentence + text[i])
            sentence = sentence[-max_len:]
            x_pred = np.zeros((1, max_len, len(words)))
        else:
            sentence = (poems[index][:max_len-1] + text[0])
            sentence = sentence[-max_len:]
            x_pred = np.zeros((1, max_len, len(words)))
            
        for j in range(4):
            for t, char in enumerate(sentence):
                x_pred[0, t, word2numF(char)] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')
            preds /= sum(preds)
            pro = np.random.choice(range(len(preds)),1,p=preds)
            next_index=int(pro.squeeze())
            next_char = num2word[next_index]
            sentence = sentence[1-max_len:] + next_char
            Mypoem[i][j]=next_char
    
    print('      无 题\n','   夏 欣 雨\n'
        ,text[0], Mypoem[0][0], Mypoem[0][1], Mypoem[0][2], Mypoem[0][3],'，\n'
         ,text[1], Mypoem[1][0], Mypoem[1][1], Mypoem[1][2], Mypoem[1][3],'。\n'
         ,text[2], Mypoem[2][0], Mypoem[2][1], Mypoem[2][2], Mypoem[2][3],'，\n'
         ,text[3], Mypoem[3][0], Mypoem[3][1], Mypoem[3][2], Mypoem[3][3],'。')

file='./poems.txt'
word2numF, num2word, words, files_content = preprocess_file(file)
poems =files_content.split(']') #list形式
poems_num = len(poems) #总数量

X = []
y = []
for i in range(len(poems)):
    for j in range(len(poems[i])-6):
        X.append(poems[i][j:j+6])
        y.append(poems[i][j+6])
        
batch_size = 2048  # 批大小
opt = Adam(lr=0.0001, decay=1e-6)  # 使用Adam优化器

model = create_model()

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_dir = os.path.join("./BestTextModel")
output_model_file = os.path.join(model_dir,
                                 "Model.h5")

callbacks = [
    ModelCheckpoint(output_model_file, save_best_only = False),
]

number_of_epoch = len(files_content)-(7)*poems_num
number_of_epoch /= batch_size 
number_of_epoch = int(number_of_epoch / 1.5)
    
history = model.fit_generator(
  generator=data_generator(),
  steps_per_epoch=batch_size,
  verbose=2,
  epochs=number_of_epoch,
  callbacks=callbacks
)