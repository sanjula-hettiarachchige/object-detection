from threading import Thread

def one(): import object_detection_edureka.py
def two(): import text_to_voice.py

Thread(target=one).start()
Thread(target=two).start()

