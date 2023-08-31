#%%
import numpy as np
import os
import itertools
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import tensorflow as tf
plt.style.use('seaborn-white')

# 사운드 파일 디렉토리 지정
directory_path = r'D:\soundclassification' 
os.chdir(directory_path)

midi_file = "./mix_sound.wav"

#0 = 캔, 1=유리, 2,3= 플라스틱
instruments = [0, 1, 2]
num_notes = 40 
sec = 2

audio =[]
inst = []
#20개의 샘플 파일, 2초당 하나씩 샘플링
for inst_idx, note in itertools.product(range(len(instruments)),range(num_notes)):
    instrument = instruments[inst_idx]
    offset = (instrument *num_notes*sec) + (note*sec)
    print('sound : {}, note : {}, offset : {}'.format(instrument , note, offset))
    y, sr = librosa.load(midi_file, sr=None, offset = offset, duration = 2.0)
    audio.append(y)
    inst.append(inst_idx)

audio_cqt = []
for y in audio:
    ret = librosa.cqt(y, sr, hop_length=1024, n_bins=24*7, bins_per_octave=24)
    ret = np.abs(ret)
    audio_cqt.append(ret)
    
for i in range(0, len((instruments)*num_notes), num_notes):
    amp_db = librosa.amplitude_to_db(np.abs(audio_cqt[i]), ref=np.max)
    librosa.display.specshow(amp_db, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f db')
    plt.title('Instrument : {}'.format(inst[i]))
    plt.tight_layout()
    plt.show()
    
audio_np= np.array(audio, np.float32)
inst_np = np.array(inst, np.int16)
cqt_np= np.array(audio_cqt, np.float32)
inst_np = np.array(inst, np.int16)

print(audio_np.shape, inst_np.shape)

for idx in range(0, len(audio_np), num_notes):
    plt.figure(figsize=(18,2))
    plt.plot(audio_np[idx])
    plt.ylim((-1,1))
    plt.show 
    
audio_mfcc = []
for y in audio:
    ret = librosa.feature.mfcc(y=y, sr=sr)
    audio_mfcc.append(ret)
        
for i in range(0, len(instruments)*num_notes, num_notes):
    amp_db = librosa.amplitude_to_db(np.abs(audio_mfcc[i]), ref=np.max)
    librosa.display.specshow(amp_db, sr=sr, x_axis='time' ,y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Instrument : {}'.format(inst[i]))
    plt.show()
    
mfcc_np = np.array(audio_mfcc, np.float32)
inst_np = np.array(inst, np.int16)

print(mfcc_np.shape, inst_np.shape)

mfcc_np = mfcc_np.reshape((120, 20*188))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(audio_np)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(mfcc_np, inst_np, test_size = 0.2)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

audio_cqt = []
for y in audio:
    ret = librosa.cqt(y, sr, hop_length=1024, n_bins=24*7, bins_per_octave=24)
    ret = np.abs(ret)
    audio_cqt.append(ret)
    
for i in range(0, len((instruments)*num_notes), num_notes):
    amp_db = librosa.amplitude_to_db(np.abs(audio_cqt[i]), ref=np.max)
    librosa.display.specshow(amp_db, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f db')
    plt.title('Instrument : {}'.format(inst[i]))
    plt.tight_layout()
    plt.show()

from tensorflow.keras.utils import to_categorical

mfcc_np= np.array(audio_mfcc, np.float32)
mfcc_np = mfcc_np.reshape((120,20*188))
mfcc_array = np.expand_dims(mfcc_np, -1)
inst_cat = to_categorical(inst_np)

train_x, test_x, train_y, test_y = train_test_split(mfcc_array,inst_cat, test_size=0.2)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense


def plot_history(history_dict):
    loss= history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(14,5))
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, loss, 'b--', label = 'train_loss')
    ax1.plot(epochs, val_loss, 'r:', label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid()
    ax1.legend()
    
    acc= history_dict['acc']
    val_acc = history_dict['val_acc']
    
    ax1 = fig.add_subplot(1,2,2)
    ax1.plot(epochs, acc, 'b--', label = 'train_accuracy')
    ax1.plot(epochs, val_acc, 'r:', label='val_accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid()
    ax1.legend()
    
    plt.show()

mfcc_np= np.array(audio_mfcc, np.float32)
mfcc_array = np.expand_dims(mfcc_np, -1)
inst_cat = to_categorical(inst_np)

train_x, test_x, train_y, test_y = train_test_split(mfcc_array,inst_cat, test_size=0.2)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten

def model_build():
    model = Sequential()
    
    input = Input(shape=(20, 188, 1))
    
    output = Conv2D(128,3,strides=1, padding='same', activation='relu')(input)
    output = MaxPool2D(pool_size=(2,2), strides=2, padding='same')(output)
    
    output = Conv2D(256,3,strides=1, padding='same', activation='relu')(output)
    output = MaxPool2D(pool_size=(2,2), strides=2, padding='same')(output)
    
    output = Conv2D(512,3,strides=1, padding='same', activation='relu')(output)
    output = MaxPool2D(pool_size=(2,2), strides=2, padding='same')(output)
    
    output = Flatten()(output)
    output = Dense(512, activation='relu')(output)
    output = Dense(256, activation='relu')(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(3, activation='softmax')(output)
    
    model = Model(inputs=[input], outputs=output)
    
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics=['acc'])
    return model

model = model_build()
model.summary()

history = model.fit(train_x, train_y, epochs=80, batch_size = 128, validation_split=0.2)

plot_history(history.history)

#%%