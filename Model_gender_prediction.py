from tensorflow.keras.models import Sequential, save_model, load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
model = load_model('./Face-Gender-Model-2.h5', compile = True)

#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])


path1=f"./py_dosya/image/hatice.jpg"
path2=f"./py_dosya/image/karanfil.jpg"
path3=f"./py_dosya/image/ali.jpg"
path4=f"./py_dosya/image/eda.jpg"

img=cv2.imread(path4)

newimg = cv2.resize(img,(128,128))
predict = np.reshape(newimg,[1,128,128,3])

classes = model.predict_classes(predict)
print (classes)


if classes[0] == 0:
    cevap = 'Kadın'
else:
    cevap = 'Erkek'
print (cevap)  

plt.imshow(img)