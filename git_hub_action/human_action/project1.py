import numpy as np
import cv2 
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras import models,layers
import mediapipe as mp
from tensorflow.keras.applications.vgg16 import VGG16


plt.style.use("ggplot")
# preprocesing the pose
mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)

#make x and y labels
datas = []
labels = []
#loop to direction of data set location
for add in glob.glob("JPEGImages/*"):
    #by spliting choose the categorie of data set we want
    s = add.split("\\")
    if s[1].startswith("j") or s[1].startswith("run") or s[1].startswith("smoking") or s[1].startswith("reading") or s[1].startswith("applauding"):
        #reading all the chosen photo of data set
        img = cv2.imread(add)
        s1 = s[1].split("_")
        #resize it to 224 because we use vgg16
        img = cv2.resize(img,(224,224))
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pose_result = pose.process(img_rgb)
        mp_drawing.draw_landmarks(img,pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #normalaize the picture
        img = img / 255.0
        datas.append(img)
        labels.append(s1[0])
    
#encode the target to one hot encoding :[0,0,1]
l_e = LabelEncoder()
int_incoding = l_e.fit_transform(labels)
one_hot_encoding = to_categorical(int_incoding)

datas = np.array(datas)
x_train , x_test, y_train , y_test = train_test_split(datas,one_hot_encoding,test_size=0.2)




#vgg16 preprocessing
base_model = VGG16(weights="imagenet", include_top=False, input_shape=datas[0].shape)
base_model.trainable = False 




#our net with transferlearning
net = models.Sequential([base_model,
                          layers.Flatten(),
                          layers.Dense(50, activation='relu'),
                          layers.BatchNormalization(),
                          layers.Dense(20, activation='relu'),
                          layers.BatchNormalization(),
                          layers.Dense(5, activation='softmax')])


net.compile(optimizer="Adam",metrics=["accuracy"],loss="categorical_crossentropy")


n = net.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15,batch_size=32)


# evlaute our loss and accuracy of net
loss, acc = net.evaluate(x_test,y_test)



net.save("action2.h5")



plt.plot(n.history['accuracy'], label = 'train accuracy')
plt.plot(n.history['val_accuracy'], label = 'test accuracy')

plt.plot(n.history['loss'], label = 'train loss')
plt.plot(n.history['val_loss'], label = 'test loss')
plt.legend(loc='best')
plt.show()

