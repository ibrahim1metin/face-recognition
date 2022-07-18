import cv2
import os
from face_recognition import load_image_file, face_encodings , compare_faces
from numba import jit
@jit
def f():
    cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
    )
        if len(face_encodings(frame))!=0:
            for knownimgn in os.listdir("C:/Users/HP/Desktop/face1/img/known/"):
                knownimg=load_image_file("C:/Users/HP/Desktop/face1/img/known/"+knownimgn)
                knownimg_e =face_encodings(knownimg)[0]
                imgenc=face_encodings(frame)[0]
                results=compare_faces([knownimg_e],imgenc)
                if results[0]:
                    print("eşleşme bulundu")
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
f()
