from gtts import gTTS
import os
import datetime
import time
from mutagen.mp3 import MP3
from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound
import glob

language = 'en'
current_dir = os.path.dirname(os.path.realpath(__file__))
path_del = current_dir+"/audio_object"
path = current_dir+"/audio_object/"
import datetime
import time



# Converts the detected classes names into an audio file which is then played out
while True:
    try:
        file = open("detected_classes.txt","r") # Opens the text file containing the detected classes
        array = (file.readlines())
        classes = (array[0])
        classes = classes.replace("\n","")
        classes = classes.replace("'","")
        classes = classes.split(",")
        pan_array = (array[1])
        pan_array = pan_array.split(",")    
        for i in range(0,len(classes)):
            direction = ""
            name = datetime.datetime.now()
            name = str(name).replace(":","")
            name = name.replace(" ","")
            name = name.replace(".","")
            text = classes[i]
            pan_level = pan_array[i]
            if float(pan_level)<0: #Determines the position of the object
                direction = ("Left")
            else:
                direction = ("Right")
            output = gTTS(text=text+direction, lang=language, slow=False) #Adds the direction to the audio file
            text = text+name
            output.save(path+text+".mp3") #Saves the audio file as an MP3
            command = path+text+".mp3"
            print(command)
            playsound(command) #Plays the audio file
            os.remove(path+text+".mp3") #Deletes the audio file after it has been played
    except:
        pass
    
