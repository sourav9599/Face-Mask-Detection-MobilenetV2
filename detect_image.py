# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:47:46 2020

@author: sourav
"""
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np

model = load_model("mask_recog1.h5")

resMap = {
        0 : 'Mask On ',
        1 : 'Mask Off '
    }

colorMap = {
        0 : (0,255,0),
        1 : (0,0,255)
    }

# def prepImg(pth):
#     return cv2.resize(pth,(224,224)).reshape(1,224,224,3)/255.0

classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
img = cv2.imread(filename)
faces = classifier.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(60, 60))
for face in faces:

    slicedImg = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
    slicedImg = cv2.cvtColor(slicedImg, cv2.COLOR_BGR2RGB)
    slicedImg = cv2.resize(slicedImg, (224, 224))
    slicedImg = img_to_array(slicedImg)
    slicedImg = np.expand_dims(slicedImg, axis=0)
    slicedImg =  preprocess_input(slicedImg)
    pred = model.predict(slicedImg)
    acc = np.max(pred*100)
    s = str(acc)
    pred = np.argmax(pred)

    cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),colorMap[pred],2)
    cv2.putText(img, resMap[pred]+s+"%",(face[0],face[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


cv2.imshow('FaceMask Detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
