import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# WAV 파일 로드
audio_file = AudioSegment.from_wav('배경소음.wav')

# 5초 단위로 자르기
duration = audio_file.duration_seconds
start_time = 74
end_time = 79 # 5초를 밀리초로 변환

# MFCC 이미지 저장 디렉토리 생성
output_dir = r"C:\MFCC_data\train\0"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while end_time <= duration :

    # 5초 단위로 자르기
    segment = audio_file[start_time * 1000:end_time * 1000]
    
    # 잘린 WAV 파일을 numpy 배열로 변환
    y = np.array(segment.get_array_of_samples(), dtype=np.float32)  # 부동소수점 형식으로 변환

    # MFCC 추출
    n_fft = 2048
    win_length = 2048
    hop_length = 1024
    n_mels = 128
    sr = 48000
    D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
    mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
    fig = plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=0.00002), sr=sr)

# 여백 최소화
    plt.tight_layout()

    # 이미지 파일 저장
    output_path = os.path.join(output_dir, f"output_{start_time}.png")
    plt.savefig(output_path)

    # 다음 5초로 이동
    start_time = end_time
    end_time += 5
    
#2) CNN 학습 코드 (수집한 데이터를 이용하여 인공지능 학습 모델 생성)
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 경로 및 하이퍼파라미터 설정
train_dir = r"C:\MFCC_data\train"
test_dir = r"C:\MFCC_data\test"
img_height = 128
img_width = 128
batch_size = 32

# ImageDataGenerator를 사용하여 데이터 전처리 및 로딩
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(img_height, img_width),
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=True)
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(img_height, img_width),
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False)
# CNN 모델 정의 / 새롭게 만듬
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_data, epochs=5, validation_data=test_data)
# 손실값과 정확도의 변화 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(loc='best')
plt.show()

#모델 저장하기
from keras.models import load_model
model.save('cnn_model.h5')

#3) 딥러닝 코드 (학습된 모델을 가지고 실시간으로 들어오는 소리의 결과 출력)
import threading
import time
import tensorflow.keras as keras
import numpy as np
from PIL import Image

import pyaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import queue
import scipy.signal as signal
import subprocess
import sys

# 클래스 이름
class_names = ['1','2','3','4','5','6','7','8','9','10','11']

# 모델 로드
model = keras.models.load_model('cnn_model.h5')

# 각각의 클래스에 대한 화면 파일
class_to_python = {
    '1': "baby.py",
    '2': "bell.py",
    '3': "car.py",
    '4': "cat.py",
    '5': "dog.py",
    '6': "siren.py",
    '7': "1.py",
    '8': "2.py",
    '9': "3.py", 
    '10': "4.py",
    '11': None
}

# 매개변수 읽기
if len(sys.argv) > 1:
    selected_classes = sys.argv[1:]
    print(selected_classes)
else:
    print("매개변수가 전달되지 않았습니다.")

# 사용자 인터페이스
for i, class_name in enumerate(selected_classes):
    class_number = int(class_name)
    print(f"어떤 클래스를 선택하시겠습니까? ({class_number})")
    for i, class_name in enumerate(selected_classes):
        if class_name in selected_classes:
            print(f"{i+1}. {class_name} (ON)")
        else:
            print(f"{i+1}. {class_name} (OFF)")
    print("0. 종료")


#오디오 변수
chunk = 1000  # 한 번에 읽을 샘플 수
format = pyaudio.paInt16  # 오디오 포맷
channels = 1  # 모노
rate = 48000  # 샘플링 레이트
record_seconds = 1  # 녹음할 시간

#MFCC 변수
n_fft = 2048
win_length = 2048
hop_length = 1024
n_mels = 128
sr = 48000

# Global 변수
shared_list = []
frames = []

# Process_1: 마이크로 5초 소리를 저장하는 프로세스
def proc_record_5s():
    global shared_list
    global frames

    # 하이패스 필터 계수
    fs = rate
    fc = 300  # 컷오프 주파수 (필요에 따라 조정 가능)
    b, a = signal.butter(4, fc / (fs / 2), 'highpass')

    while True:
        frame = []
        for i in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frame.append(np.fromstring(data,dtype=np.int16))
        frame = np.hstack(frame)

        # 하이패스 필터 적용
        frame_filtered = signal.lfilter(b, a, frame)
        
        frames.append(frame_filtered)
        if len(frames) == 5:
            print("Complete 5s recording...")
            shared_list.append([num for elem in frames for num in elem])
            frames.append(frames.pop(0))
            frames.pop()

# Process_2: 학습 코드 반복 실행
def proc_inference():
    global shared_list
    while True:
        if shared_list:
            print("Start inference...")
            y = np.asarray(shared_list[0])
            y_float = y.astype(np.float32)
            shared_list = []

            D = np.abs(librosa.stft(y_float, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
            mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
            
            # MelSpectrum.png 파일로 저장하는 부분
            fig_for_save = plt.figure()
            librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=0.00002), sr=sr, hop_length=hop_length)
            plt.tight_layout(pad=0)
            plt.close(fig_for_save)
            fig_for_save.savefig('temp.png', pad_inches=0)

            image = Image.open('temp.png').convert('RGB')
            image = image.resize((128, 128))
            image_array = np.array(image)
            image_array = image_array / 255.0

        # 예측 함수
        def predict(image_path, selected_classes):
            try:
                # 이미지 불러오기
                image = Image.open(image_path).convert('RGB')
            except:
                print("이미지 파일을 찾을 수 없습니다.")
                return

            # 이미지 크기 조절
            image = image.resize((128, 128))

            # 이미지 배열로 변환
            image_array = np.array(image)

            # 이미지 스케일링
            image_array = image_array / 255.0

            # 클래스 예측
            predictions = model.predict(np.array([image_array]))

            # 선택된 클래스에 대한 예측 확률 추출
            selected_indices = [class_names.index(class_name) for class_name in selected_classes]
            selected_predictions = predictions[0][selected_indices]

            # Softmax 함수 적용
            exp_predictions = np.exp(selected_predictions)
            softmax_predictions = exp_predictions / np.sum(exp_predictions)

            # 예측 확률 기준 내림차순으로 클래스 정렬
            sorted_classes = [class_names[i] for i in np.argsort(selected_predictions)[::-1]]

            # 선택된 클래스가 2개 이하일 때는 선택된 클래스만 출력
            if len(sorted_classes) <= 2:
                sorted_classes = selected_classes

            # 상위 3개 클래스(0.3 이상) 이름과 예측 확률 출력
            print("상위 3개 클래스:")
            for class_name in sorted_classes:
                prediction = softmax_predictions[selected_classes.index(class_name)]
                if prediction >= 0.3:
                    print(f"{class_name}: {prediction}")

                # 예측된 클래스에 해당하는 이미지 출력
            for class_name in sorted_classes:
                if class_name in class_to_python:
                    python_file = class_to_python[class_name]
                    try:
                        # 파이썬 파일 실행
                        subprocess.run(['python', python_file])

                    except:
                        print(f"{class_name} 파일을 실행할 수 없습니다.")

        image_path = 'temp.png' # 내가 원하는 이미지 파일 경로 지정

        try:
            predict(image_path, selected_classes)
            print("선택된 클래스:")
            for class_name in selected_classes:
                print(class_name)
        except:
            print("끝")

# main
def main():
    global p
    global stream

    # PyAudio 객체 생성
    p = pyaudio.PyAudio()

    # 마이크 장치 정보 확인
    mic_device_index = None
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if 'mic' in dev_info['name'].lower():
            mic_device_index = dev_info['index']
            break

    if mic_device_index is None:
        print('Mic device not found')
        exit()

    # 스트림 열기
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=mic_device_index,
                    frames_per_buffer=chunk)

    p1 = Thread(target=proc_record_5s)
    p2 = Thread(target=proc_inference)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    # 녹음 종료
    stream.stop_stream()
    stream.close()
    # PyAudio 객체 종료
    p.terminate()

if __name__ == '__main__':
    main()

# 스레드 생성 및 실행
thread_1 = threading.Thread(target=main)
thread_2 = threading.Thread(target=proc_record_5s)
thread_3 = threading.Thread(target=proc_inference)
thread_1.start()
thread_2.start()
thread_3.start()


#4) 소리 추가 학습 (실시간으로 들어오는 소리를 변조를 통해 데이터로 만든 후 재학습)

import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy import misc
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#pyaudio 라이브러리를 사용하여 마이크에서 입력된 소리를 wav 파일로 저장한 후 mfcc로만든 코드 (전체적)->그다음 학습을 시키면 될것 같은댕

#PyAudio 라이브러리를 사용하여 마이크에서 입력된 소리를 WAV 파일로 저장

#녹음할 설정값들을 지정합니다.
chunk = 1024  # 한 번에 읽을 샘플 수
format = pyaudio.paInt16  # 오디오 포맷
channels = 1  # 모노
rate = 48000  # 샘플링 레이트
record_seconds = 5  # 녹음할 시간

# PyAudio 객체 생성
p = pyaudio.PyAudio()

# 마이크 장치 정보 확인
mic_device_index = None
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if 'mic' in dev_info['name'].lower():
        mic_device_index = dev_info['index']
        break

if mic_device_index is None:
    print('Mic device not found')
    exit()

# 스트림 열기
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=mic_device_index,
                frames_per_buffer=chunk)

# 녹음 시작
frames = []
for i in range(0, int(rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)

# 녹음 종료
stream.stop_stream()
stream.close()

# PyAudio 객체 종료
p.terminate()

# WAV 파일로 저장
segment_length = rate  # 5초 길이
for i, segment in enumerate(range(0, len(frames), segment_length)):
    filename = f"recorded_audio!_0.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames[segment:segment+segment_length]))
    wf.close()

print('Recording complete.')

#변조 파트
n_fft = 2048
win_length = 2048
hop_length = 1024
n_mels = 128

def make_melspectrum_and_save(y, sr, n_steps, path_with_name):
    y_pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    D = np.abs(librosa.stft(y_pitch, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
    mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)
    fig = plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=0.00002), sr=sr, hop_length=hop_length)
    fig.savefig(path_with_name, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    y, sr = librosa.load('./recorded_audio!_0.wav') # file load
    shift_factors = np.linspace(-0.5, 0.5, num=420)

    # save 60 samples to the test directory
    test_dir = r"C:\MFCC_data\test\new1"
    for idx, shift_factor in enumerate(shift_factors[:60]):
        melspec_file_name = 'test_data{:04d}.png'.format(idx)
        make_melspectrum_and_save(y=y, sr=sr, n_steps=shift_factor, path_with_name=os.path.join(test_dir, melspec_file_name))

    # save 360 samples to the train directory
    train_dir = r"C:\MFCC_data\train\new1"
    for idx, shift_factor in enumerate(shift_factors[60:-1]):
        melspec_file_name = 'train_data{:04d}.png'.format(idx)
        make_melspectrum_and_save(y=y, sr=sr, n_steps=shift_factor, path_with_name=os.path.join(train_dir, melspec_file_name))

# 이미지 경로 및 하이퍼파라미터 설정
train_dir = 'C:/MFCC_data/train'
test_dir = 'C:/MFCC_data/test'
img_height = 128
img_width = 128
batch_size = 32

# ImageDataGenerator를 사용하여 데이터 전처리 및 로딩
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(img_height, img_width),
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=True)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(img_height, img_width),
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False)

# CNN 모델 정의 / 새롭게 만듬
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_data, epochs=13, validation_data=test_data)

#모델 저장하기
from keras.models import load_model
model.save('cnn_model.h5')

#5) 결과 화면 출력 코드 - (bell, car, siren, cat, dog, baby, new1, new2, new3, new4)

import tkinter as tk
from tkinter import font

root = tk.Tk()
root.geometry("800x480") # 화면 크기를 800x480 픽셀로 고정합니다.

# 이미지를 불러옵니다.
photo = tk.PhotoImage(file="bell.jpg").subsample(3) # 이미지 크기를 100x100 픽셀로 조정합니다.

# 이미지를 표시하기 위해 레이블 위젯을 생성합니다.
label = tk.Label(root, image=photo, anchor="center")
label.pack(pady=20)

# 텍스트를 표시하기 위해 레이블 위젯을 생성합니다.
text_label = tk.Label(root, text="초인종 소리가 감지되었습니다", anchor="center", font=("Helvetica", 32, "bold"))
text_label.pack()

root.after(3000, root.destroy)  # 3초 후에 화면이 사라지도록 설정합니다.

root.mainloop()

from tkinter import *
import tkinter as tk
import tkinter.ttk
import tkinter.font
import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import time

# 클래스 이름 (전역변수)
class_names = ['baby', 'bell', 'car', 'cat', 'dog', 'siren', 'new1', 'new2', 'new3', 'new4', 'noise']
class_names_display = ["아기 울음소리", "초인종 소리", "차 경적 소리", "고양이 소리", "개 짖는 소리", "경보음 소리", "new1", "new2", "new3", "new4", "배경소음(주변 소음이 있을 시 선택하세요.)"]
global checkbox_values
checkbox_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
global selected_classes_numbers
selected_classes_numbers = []

class FirstPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill='both', expand=True,)
        
        frame = tk.Frame(self)  # 체크박스를 포함하는 부모 위젯
        frame.configure(bg='pink')
        frame.place(relx=0.5, rely=0.42, anchor='center')  # 중앙에 배치

        self.CheckVar1 = tk.IntVar()
        self.CheckVar2 = tk.IntVar()
        self.CheckVar3 = tk.IntVar()
        self.CheckVar4 = tk.IntVar()
        self.CheckVar5 = tk.IntVar()
        self.CheckVar6 = tk.IntVar()
        self.CheckVar7 = tk.IntVar()
        self.CheckVar8 = tk.IntVar()
        self.CheckVar9 = tk.IntVar()
        self.CheckVar10 = tk.IntVar()
        self.CheckVar11 = tk.IntVar()
        self.CheckVar1.set(checkbox_values[0])
        self.CheckVar2.set(checkbox_values[1])
        self.CheckVar3.set(checkbox_values[2])
        self.CheckVar4.set(checkbox_values[3])
        self.CheckVar5.set(checkbox_values[4])
        self.CheckVar6.set(checkbox_values[5])
        self.CheckVar7.set(checkbox_values[6])
        self.CheckVar8.set(checkbox_values[7])
        self.CheckVar9.set(checkbox_values[8])
        self.CheckVar10.set(checkbox_values[9])
        self.CheckVar11.set(checkbox_values[10])

        # class_names의 순서와 아래 체크박스의 순서가 일치해야 함
        checkboxes = [
            (class_names_display[0], self.CheckVar1),
            (class_names_display[1], self.CheckVar2),
            (class_names_display[2], self.CheckVar3),
            (class_names_display[3], self.CheckVar4),
            (class_names_display[4], self.CheckVar5),
            (class_names_display[5], self.CheckVar6),
            (class_names_display[6], self.CheckVar7),
            (class_names_display[7], self.CheckVar8),
            (class_names_display[8], self.CheckVar9),
            (class_names_display[9], self.CheckVar10),
            (class_names_display[10], self.CheckVar11),
        ]

        for text, var in checkboxes:
            c = tk.Checkbutton(frame, text=text, variable=var, font=(None, 14))
            c.pack(anchor='w', padx=10, pady=3)

        button_frame = tk.Frame(self)
        button_frame.pack(side='bottom', pady=10, padx=10)

        button = tk.Button(button_frame, text="선택완료", font=(None, 25), command=self.get_checkbox_values)
        button.pack(side='right')

    def refresh(self):
        self.destroy()
        self.__init__()

    def get_checkbox_values(self):
        
        global checkbox_values
        checkbox_values = [
            self.CheckVar1.get(),
            self.CheckVar2.get(),
            self.CheckVar3.get(),
            self.CheckVar4.get(),
            self.CheckVar5.get(),
            self.CheckVar6.get(),
            self.CheckVar7.get(),
            self.CheckVar8.get(),
            self.CheckVar9.get(),
            self.CheckVar10.get(),
            self.CheckVar11.get()
        ]
        
        selected_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        selected_classes = [class_name for value, class_name in zip(checkbox_values, selected_classes) if value == 1]

        print("선택된 클래스:")

        selected_classes_numbers.extend(str(class_num) for class_num in selected_classes)
        print(selected_classes_numbers)

        #다음 페이지 이동
        self.master.show_second_page()

    def get_property(self):
        return selected_classes_numbers

class SecondPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        
        # 버튼을 담을 프레임을 생성
        button_frame = tk.Frame(self)
        button_frame.pack(side='bottom', pady=10, padx=10)
        
        tk.Label(self, text="실행중...", font=(None, 35)).pack(pady=120)
        
        tk.Button(button_frame, text="목록으로", font=(None, 25), command=master.show_first_page).pack(side='left', anchor='sw', padx=(10, 300), pady=(0, 20))
        
        tk.Button(button_frame, text="내 소리 추가", font=(None, 25), command=master.show_third_page).pack(side='right', anchor='se', padx=(0, 10), pady=(0, 20))
        
        tk.Button(button_frame, text="실행", font=(None, 25), command=self.run_final).pack(side='right', anchor='se', padx=(0, 10), pady=(0, 20))
    
        print(FirstPage.get_property(self))

    #model 실행
    def run_final(self):
        process = subprocess.Popen(['python', 'final.py'] + FirstPage.get_property(self))
        time.sleep(100)  # 100초 동안 대기
        process.terminate()  # 모델 실행 프로세스 종료

import subprocess

class ThirdPage(tk.Frame):
    
    def __init__(self, master):
        super().__init__(master)
              
        tk.Label(self, text="원하는 파일의 버튼을 누르면 5초씩,\n\n녹음이 진행됩니다.", font=(None, 32)).pack(pady=50)

        tk.Button(self, text="newfile1", font=(None, 18), command=self.new1).place(x=100, y=330)
        self.t6 = tk.Entry(self, font=(None, 18))
        self.t6.place(x=100, y=400, width=100, height=30)
        tk.Button(self, text="newfile2", font=(None, 18), command=self.new2).place(x=250, y=330)
        self.t7 = tk.Entry(self, font=(None, 18))
        self.t7.place(x=250, y=400, width=100, height=30)
        tk.Button(self, text="newfile3", font=(None, 18), command=self.new3).place(x=400, y=330)
        self.t8 = tk.Entry(self, font=(None, 18))
        self.t8.place(x=400, y=400, width=100, height=30)
        tk.Button(self, text="newfile4", font=(None, 18), command=self.new4).place(x=550, y=330)
        self.t9 = tk.Entry(self, font=(None, 18))
        self.t9.place(x=550, y=400, width=100, height=30)
    
    def new1(self):
        #전역변수 초기화
        global selected_classes_numbers
        selected_classes_numbers = []
         
        class_names_display[6] = self.t6.get()
        self.master.show_fourth_page()
        subprocess.Popen(["python", "new1.py"])  # new1.py 파일 실행

    def new2(self):
        #전역변수 초기화
        global selected_classes_numbers
        selected_classes_numbers = []
        
        class_names_display[7] = self.t7.get()
        self.master.show_fourth_page()
        subprocess.Popen(["python", "new2.py"])  # new2.py 파일 실행
 
    def new3(self):
        #전역변수 초기화
        global selected_classes_numbers
        selected_classes_numbers = []
        
        class_names_display[8] = self.t8.get()
        self.master.show_fourth_page()
        subprocess.Popen(["python", "new3.py"])  # new3.py 파일 실행

    def new4(self):
        #전역변수 초기화
        global selected_classes_numbers
        selected_classes_numbers = []
        
        class_names_display[9] = self.t9.get()
        self.master.show_fourth_page()
        subprocess.Popen(["python", "new4.py"])  # new4.py 파일 실행

class FourthPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="녹음중...", font=(None, 35)).pack(pady=100)
        tk.Button(self, text="녹음완료", font=(None, 32), command=master.show_fifth_page).pack(pady=(110, 0))

class FifthPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        tk.Label(self, text="학습중...", font=(None, 35)).pack(pady=80)
        tk.Button(self, text="Home", font=(None, 25), command=master.show_first_page).pack(side='right', anchor='se', padx=(0, 10), pady=(0, 20))     
        tk.Button(self, text="학습시작", font=(None, 25), command=self.cnn_model).pack(side='right', anchor='se', padx=(0, 10), pady=(0, 20))

    #model 실행
    def cnn_model(self):
        cnn = subprocess.Popen(['python', 'cnn_model.py'] + FirstPage.get_property(self))
        time.sleep(240)  # 240초 동안 대기
        cnn.terminate()  # 모델 실행 프로세스 종료

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('800x480')
        self.first_page = FirstPage(self)
        self.second_page = SecondPage(self)
        self.third_page = ThirdPage(self)
        self.fourth_page = FourthPage(self)
        self.fifth_page = FifthPage(self)
        self.first_page.pack(fill="both", expand=True)

    def show_first_page(self):
        self.first_page.__init__(self)
        self.first_page.pack(fill="both", expand=True)
        self.second_page.pack_forget()
        self.third_page.pack_forget()
        self.fourth_page.pack_forget()
        self.fifth_page.pack_forget()

    def show_second_page(self):
        self.first_page.pack_forget()
        self.second_page.pack(fill="both", expand=True)
        self.third_page.pack_forget()
        self.fourth_page.pack_forget()
        self.fifth_page.pack_forget()

    def show_third_page(self):
        self.first_page.pack_forget()
        self.second_page.pack_forget()
        self.third_page.pack(fill="both", expand=True)
        self.fourth_page.pack_forget()
        self.fifth_page.pack_forget()

    def show_fourth_page(self):
        self.first_page.pack_forget()
        self.second_page.pack_forget()
        self.third_page.pack_forget()
        self.fourth_page.pack(fill="both", expand=True)
        self.fifth_page.pack_forget()
        self.after(210000, self.show_fifth_page)  #  210초 후 학습중 화면으로 전환됨

    def show_fifth_page(self):
        self.first_page.pack_forget()
        self.second_page.pack_forget()
        self.third_page.pack_forget()
        self.fourth_page.pack_forget()
        self.fifth_page.pack(fill="both", expand=True)
        self.after(260000, self.show_first_page)  # 260초 후 처음 화면으로 전환됨

if __name__ == '__main__':
    app = MainApplication()
    app.mainloop()