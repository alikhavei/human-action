from tkinter import ttk
from tkinter import *
from tkinter import filedialog
import cv2
from keras.models import load_model
import mediapipe as mp
import numpy as np

#loading the trained model
model = load_model("action.h5")

#preprocessing the pose
mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)


#making tkinter gui
root = Tk()
root.title("human_action")
root.geometry("800x600")
def img():
    name = filedialog.askopenfilename(filetypes=[('JPG files', '*.jpg'),
                                                  ('PNG files', '*.png'),],
                                       title="select img"

                                                  
                                        )


    img = cv2.imread(name)

    img = cv2.resize(img,(224,224))

    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    pose_result = pose.process(rgb_img)
    mp_drawing.draw_landmarks(img,pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('af',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = img / 255.0

    img = np.array([img])



    predict = model.predict(img)[0]

    out = np.argmax(predict)



    if out == 0:
        print("clap")

    elif out == 3:
        print("run")

    elif out == 2:
        print("reading")


    elif out == 4:
        print("smoke")


    elif out == 1:
        print("jump")





def mp4():

    name = filedialog.askopenfilename(filetypes=[('video files', '*.mp4')],
                                       title="select mp4"

                                                  
                                        )

    cap = cv2.VideoCapture(name)


    while True :
        _ , frame = cap.read()

        frame = cv2.resize(frame,(224,224))
        

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        pose_result = pose.process(frame_rgb)



        mp_drawing.draw_landmarks(frame,pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)



        img = frame / 255.0

        img = np.array([img])

    

        predict = model.predict(img)[0]

        out = np.argmax(predict)



        if out == 0:
            text = "clap"

        elif out == 3:
            text = "run"

        elif out == 2:
            text = "reading"


        elif out == 4:
            text = "smoke"


        elif out == 1:
            text = "jump"

        cv2.putText(frame,text,(20,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord("x"):
            break
        

    cv2.destroyAllWindows()


def cam():

    cap = cv2.VideoCapture(0)


    while True :
        _ , frame = cap.read()

        frame = cv2.resize(frame,(224,224))
            

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
        pose_result = pose.process(frame_rgb)



        mp_drawing.draw_landmarks(frame,pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)



        img = frame / 255.0

        img = np.array([img])

        

        predict = model.predict(img)[0]

        out = np.argmax(predict)



        if out == 0:
            text = "clap"

        elif out == 3:
            text = "run"

        elif out == 2:
            text = "reading"


        elif out == 4:
            text = "smoke"


        elif out == 1:
            text = "jump"

        cv2.putText(frame,text,(20,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord("x"):
            break
            

    cv2.destroyAllWindows()

 


but1 = ttk.Button(root,text="import image",command=img)
but1.pack(ipadx=50,ipady=50)
#import a video to prediction
but1 = ttk.Button(root,text="import video",command=mp4)
but1.pack(ipadx=50,ipady=50,pady=100,padx=100)
#access the camera
but2 = ttk.Button(root,text="cammera",command=cam)
but2.pack(ipadx=50,ipady=50)
mainloop()

