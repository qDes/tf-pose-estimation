import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import imutils

import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        
        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Rotate", width=25, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        
        #btn
        self.btn_add = tkinter.Button(window, text = "++", width = 25, command = self.adding)
        self.btn_add.pack(anchor = tkinter.CENTER, expand = True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.deg = 0
        self.update()
        self.window.mainloop()

    def snapshot(self):
        '''
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        '''
        self.deg += 90

    def adding(self):
        self.vid.y += 10

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame(self.deg)
            
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)


         self.x = 10
         self.y = 100

         self.counter = 3
         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
         self.e = TfPoseEstimator(get_graph_path("mobilenet_thin") , target_size=(320,192)) 


    def get_frame(self, deg):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                frame = imutils.rotate_bound(frame, deg)
                humans = self.e.inference(frame, resize_to_default=(self.width and self.height),upsample_size = 4.0)
                frame,neck_y = TfPoseEstimator.draw_humans(frame, humans, imgcopy = False)
                #print(self.y)
                if np.abs(neck_y-self.y) < 100 and self.y < neck_y:
                    self.y = neck_y#np.abs(neck_y-self.y)
                if self.y > 400:
                    self.y = 100
                    if self.counter > 0:
                        self.counter -=1
                if self.counter == 0:
                    cv2.putText(frame,'Pososee',(200,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
                else:
                    cv2.putText(frame,'Make '+str(self.counter) + ' squats',(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

                #circle drawing
                cv2.circle(frame,(self.x,self.y),5,(0,255,0),-1)
                

                try:
                    #print(type(humans[0]))
                    pass
                except:
                    pass
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
     # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    App(tkinter.Tk(), "Test ebat'") 
