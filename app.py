from tensorflow.keras.models import load_model
import librosa
import numpy as np

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

model=load_model('model_2.h5')

sample_path='sample/5 Oct, 17.48_.wav'
data, sample_rate = librosa.load(sample_path, duration=2.5, offset=0.6)
sample_features=extract_features(data,sample_rate)

sample_features=sample_features.reshape(1,162,1)
sample_features.shape
result=model.predict(sample_features)

sentiment_key={0:'angry',1:'calm',2:'disgust',3:'fear',4:'happy',5:'neutral',6:'sad',7:'surprise'}

print('Emotion = ',sentiment_key[result.argmax(axis=1)[0]])