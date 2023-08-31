
#%%
import librosa
import numpy as np
import os
import tensorflow as tf
import pyaudio
import wave

# 사운드 파일 디렉토리 지정
directory_path = r'D:\soundclassification'  # WAV 파일이 있는 디렉토리 경로
os.chdir(directory_path)

# 학습된 모델 불러오기
model = tf.keras.models.load_model('Audio_classifier.h5')

# PyAudio를 사용하여 오디오 스트림 열기
p = pyaudio.PyAudio()

# 오디오 스트림 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS =2

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* 녹음 시작")

while True:
    frames = []
    
    # 오디오 스트림에서 데이터 읽기
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    # 읽은 데이터를 WAV 파일로 저장 (선택 사항)
    wf = wave.open("temp_audio.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # 음성 데이터 로드 및 전처리
    def preprocess_audio(audio_file_path):
        # 음성 파일 로드
        y, sr = librosa.load(audio_file_path, sr=None, duration=2.0)
        
        # MFCC 특성 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        
        N = 173
        mfcc = librosa.util.fix_length(mfcc, N, axis=1)
        mfcc = mfcc.reshape((1, 20, N, 1))

        return mfcc
    
    # 저장한 WAV 파일을 모델에 입력으로 전달하여 예측
    input_data = preprocess_audio("temp_audio.wav")
    predicted_probabilities = model.predict(input_data)
    predicted_class = np.argmax(predicted_probabilities)
    
    # 예측 결과 출력
    if predicted_class == 0:
        print("No sound")
    elif predicted_class == 1:
        print("Can sound")
    elif predicted_class == 2:
        print("Glass sound")
    elif predicted_class == 3:
        print("Plastic sound")
    else:
        print("other sound")
# %%
