#voice analysis
import librosa
import soundfile
import os,glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def extract_feature(file_name,mfcc,chroma,mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X=sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfcc=np.mean(librosa.featute.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
            result=np.hstack((result,chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectogram(X,sr=sample_rate).T,axis=0)
            result=np.hstack((result,mel))
    return result
emotions={'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
#emotions to observe
observed_emotions=['calm','happy','fearful','disgust']
#Load the data and extract feaures of each sound file
def load_data(test_size=0.2):
   x,y=[],[]
   for file in glob.glob("C:/Users/ANANTH/Desktop/test.wav"):
      file_name=os.path.basename(file)
      emotion=emotions[file_name.split("-")[0]]
      if emotion not in observed_emotions:
          continue
      feature=extract_feature(file,mfcc=True,chroma=True,mel=True)
      x.append(feature)
      y.append(emotion)
   return train_test_split(np.array(x),y,test_size=test_size,random_state=9)


file="C:/Users/ANANTH/Desktop/test.wav"
feature=extract_feature(file,mfcc=True,chroma=True,mel=True)
#split the data set
x_train,x_test,y_train,y_test=load_data(test_size=0.25)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
# Get the shape of training and testing data set
print((x_train.shape[0],x_test.shape[0]))

# Get the number of features extracted
print('Features extracted: {x_train.shape[1]}')

# initialize the multilayer perceptron calssifier
model=MLPClassifier(alpha=0.01,batch_size=256,epsilon=1e-08,hidden_layer_sizes=(300,),learning_rate='adaptive',max_iter=500)

#train the model
model.fit(x_train,y_train)
#predict for testset
y_pred=model.predict(x_test)
y_pre=model.predict([feature])
print(y_pre)
time.sleep(2) 

# calculate the accuracy od our model
accuracy=accuracy_score(y_true=y_test,y_pred=y_pred)
## print th accuracy
print("Accuracy:{:.2f}%".format(accuracy*100))
print(y_pred)

  
            