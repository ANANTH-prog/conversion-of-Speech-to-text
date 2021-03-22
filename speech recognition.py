import speech_recognition as sr
import pandas as pd
import numpy as np

r = sr.Recognizer()

audio_file = sr.AudioFile("C:/Users/ANANTH/Desktop/preamble.wav")

with audio_file as source:
    r.adjust_for_ambient_noise(source)
    audio = r.record(source)
result = r.recognize_google(audio)

with open("C:/Users/ANANTH/Desktop/preamble1.txt",mode ="w") as file:
    file.write("Recognized text:")
    file.write("\n")
    file.write(result)
    print("Hurray! conversion is complete")
    

##
from textblob import TextBlob
import nltk
nltk.download("punkt")
text =open("C:/Users/ANANTH/Desktop/test.txt", "r")
print(text.read())
##create a textblob object
obj=TextBlob(str(text))
#this returns a value between -1 and 1
sentiment=obj.sentiment.polarity
print(sentiment)
#doing sentiment analysis split them in to positive, Negative, Neutral
if sentiment == 0:
    print("the text is neutral")
elif sentiment > 0:
    print("text is positive")
else:
    print("the text is negative")
    

