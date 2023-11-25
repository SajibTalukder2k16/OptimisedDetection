# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:57:40 2023

@author: Sajib
"""
import statistics as st
import queue

import math
import sys
import cv2
import random
import time
import math
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QSizePolicy,QProgressBar
status = "Status: "
person1_status = ""
person2_status = ""
button_style = """
    QPushButton {
        background-color: blue;
        color: white;
        font: bold 18px;
        height: 32px; /* Set the height */
        width: 48px; /* Set the width */
        border-radius: 10px; /* Set the border radius */
    }
"""

import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces=2)
#################################


# Load Yolo
# =============================================================================
# net = cv2.dnn.readNet(r"C:\Users\12028\Desktop\yolov3.weights", r"C:\Users\12028\Desktop\yolov3.cfg")
# classes = []
# with open(r"C:\Users\12028\Desktop\coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))
# =============================================================================
heart_attack1 = 0
heart_attack2 = 0
# Loading camera

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_tracking_confidence=0.5, min_detection_confidence=0.5, model_complexity=0, smooth_landmarks=True)
ww = 480
hh = 480

##################################

# Initialize VideoCapture to read the input video file
#input_video_path = r"C:/Users/saifu/Box/Summer 2023/momentai/video_library/Seizure_ppp/SaveTube.io-Full Blown Grand Mal Seizure-(heighp).mp4"

# Create two priority queues, one for max values and one for min values
qsize = 20
max_queue = queue.PriorityQueue(maxsize=qsize+1)
min_queue = queue.PriorityQueue(maxsize=qsize+1)


max_queue1 = queue.PriorityQueue(maxsize=qsize+1)
min_queue1 = queue.PriorityQueue(maxsize=qsize+1)

max_queue2 = queue.PriorityQueue(maxsize=qsize+1)
min_queue2 = queue.PriorityQueue(maxsize=qsize+1)

left_blink=0 
right_blink=0

sleep = 0
drowsy = 0
active = 0
status=""
yawn_count = 0
distraction =0
heart_attack=0
centroidx = [0,0,0]
centroidy =[0,0,0]
centroidz=[0,0,0]
#centroidz = []
CA= 0
CA_signal = 0
dummyx=0
dummyy=0
sdvxlist=[]
sdvylist=[]
XList = []
YList = []
XAvgList = []
YAvgList = []
c=0


sleep1 = 0
drowsy1 = 0
active1 = 0
status1 =""
yawn_count1 = 0
distraction1 =0
heart_attack1=0
centroidx1 = [0,0,0]
centroidy1 =[0,0,0]
centroidz1=[0,0,0]
CA1= 0
CA_signal1 = 0
dummyx1=0
dummyy1=0
sdvxlist1=[]
sdvylist1=[]
XList1 = []
YList1 = []
XAvgList1 = []
YAvgList1 = []
c1=0


## person2
sleep2 = 0
drowsy2 = 0
active2 = 0
status2 =""
yawn_count2 = 0
distraction2 =0
heart_attack2=0
centroidx2 = [0,0,0]
centroidy2 =[0,0,0]
centroidz2=[0,0,0]
CA2= 0
CA_signal2 = 0
dummyx2=0
dummyy2=0
sdvxlist2=[]
sdvylist2=[]
XList2 = []
YList2= []
XAvgList2 = []
YAvgList2 = []
c2=0
manualThreshold = .15
v0=0
x0=0


progressbar1_value = 0;
progressbar2_value = 0;


# Define codec and VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'time_fatigue_wfm.mp4'

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
number_of_faces=5
def compute(ptA,ptB):
    dist = math.sqrt((ptA[0]-ptB[0])**2+ (ptA[1]-ptB[1])**2)
 
    return dist

def blinked(a,b,c,d,e,f):
            up = compute(b,d) + compute(c,e)
            down = compute(a,f)
            ratio = up/(2.0*down)
            return ratio
        	#Checking if it is blinked
            #print("Eye_Ratio = " + str(ratio))

def detect(ratio,threshhold):
            if(ratio<threshhold):
                #print('sleeping')
                return 0
            elif(ratio>threshhold and ratio<=1.05*threshhold):
                return 1
            else:
                return 2



def velocities(centroidx1,centroidy1,centroidz1,frames_observed)  :
    frames_observed=frames_observed/1000
    vx=np.gradient(centroidx1, frames_observed)
    vy=np.gradient(centroidy1, frames_observed)
    vz=np.gradient(centroidz1, frames_observed) 
    vx_mean=np.mean(vx)
    vy_mean=np.mean(vy)
    vz_mean=np.mean(vz)
    v_magnitude=math.sqrt(vx_mean**2+vy_mean**2+vz_mean**2) 
    return vx,vy,vz,v_magnitude  

def acceleration(vx,vy,vz,frames_observed)   :
    frames_observed=frames_observed/1000
    ax=np.gradient(vx, frames_observed)
    ay=np.gradient(vy, frames_observed)
    az=np.gradient(vz, frames_observed)
    ax_mean=np.mean(ax)
    ay_mean=np.mean(ay)
    az_mean=np.mean(az)
    a_magnitude=math.sqrt(ax_mean**2+ay_mean**2+az_mean**2) 
    return ax,ay,az,a_magnitude


def jerk(ax,ay,az,frames_observed)   :
    frames_observed=frames_observed/1000
    jx=np.gradient(ax, frames_observed)
    jy=np.gradient(ay, frames_observed)
    jz=np.gradient(az, frames_observed)
    jx_mean=np.mean(ax)
    jy_mean=np.mean(ay)
    jz_mean=np.mean(az)
    j_magnitude=math.sqrt(jx_mean**2+jy_mean**2+jz_mean**2) 
    return jx,jy,jz,j_magnitude




                
def yawn(g,h,i,j,k,l,m,n):
    d1 = compute(g,j)
    d2 = compute(h,k)
    d3 = compute(i,l)
    d4 = compute(m,n)
    yawn_d = (d1+d2+d3)/(3*d4)
    
    #print("d1 =" + str(d1) + "d2 =" + str(d2) + "d3 =" + str(d3) +"d4 =" + str(d4))
    
    
    #print("Yawn_ratio = "+ str(yawn_d))
    
    if(yawn_d>1):
        return 3
    else:
        return 0
    


image_sources_shared_default = ['shared/default/machine.png','shared/default/person1.png','shared/default/person2.png','shared/default/phone.png']
image_sources_shared_active = ['shared/active/machine.png','shared/active/person1.png','shared/active/person2.png','shared/active/phone.png']
image_sources_active = ['active/sleepy.png','active/yawning.png','active/heartAttack.png','active/epilepsy.png']
image_sources_default = ['default/sleepy.png','default/yawning.png','default/heartAttack.png','default/epilepsy.png']
class CameraPreview(QMainWindow):
    def __init__(self):
        global image_sources_default,image_sources_active;
        super().__init__()

        self.setWindowTitle("Camera Preview")
        self.setGeometry(100, 100, 800, 600)
        

        # Create a QWidget to contain both images with a QHBoxLayout
        self.image_container = QWidget(self)
        self.image_container_layout = QHBoxLayout()
        self.image_container.setLayout(self.image_container_layout)

        # Create a QLabel for the left image
        self.top_left_image = QLabel()
        self.image_container_layout.addWidget(self.top_left_image)

        # Create a spacer to push the right image to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_container_layout.addWidget(spacer)

        # Create a QLabel for the middle image
        self.middle_image = QLabel()
        self.image_container_layout.addWidget(self.middle_image)

        # Create a spacer to push the right image to the right
        spacer2 = QWidget()
        spacer2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_container_layout.addWidget(spacer2)

        # Create a QLabel for the right image
        self.top_right_image = QLabel()
        self.image_container_layout.addWidget(self.top_right_image)

        # Load and set the images for the top left, middle, and top right labels with a size of 60x60 pixels
        self.load_images()

        # Create a QLabel to display the camera feed
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Create Start, Stop, and Quit buttons
        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)
        #self.quit_button = QPushButton("Quit", self)

        # Apply stylesheets to buttons (color and font)
        self.start_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)
       # self.quit_button.setStyleSheet(button_style)

        # Create layouts for buttons
        button_layout1 = QHBoxLayout()
        button_layout1.addWidget(self.start_button)
        button_layout1.addWidget(self.stop_button)

        button_layout2 = QVBoxLayout()
        #button_layout2.addWidget(self.quit_button)

        # Create a container widget for buttons
        button_container = QWidget()
        button_container.setLayout(button_layout1)
        

        # Create two boxes at the bottom of the window with an outer frame
        self.bottom_box1 = QWidget(self)
        self.bottom_box1.setStyleSheet("border-radius: 5px;background-color: white")  # Add an outer frame
        self.bottom_box1_layout = QVBoxLayout()
        self.bottom_box1.setLayout(self.bottom_box1_layout)

        self.bottom_box2 = QWidget(self)
        self.bottom_box2.setStyleSheet("border-radius: 5px; background-color: white")  # Add an outer frame
        self.bottom_box2_layout = QVBoxLayout()
        self.bottom_box2.setLayout(self.bottom_box2_layout)

        # Create labels for each box
        self.label1_1 = QLabel("Person 1       ID: MomentAI_7865", self)
        self.label1_1.setAlignment(Qt.AlignCenter)
        
        #self.label1_2 = QLabel("ID: MomentAI_7865", self)
        #self.label1_2.setAlignment(Qt.AlignCenter)
        
        #self.label1_3 = QLabel(status+person2_status, self)
        #self.label1_3.setAlignment(Qt.AlignCenter)
        
        self.label2_1 = QLabel("Person 2        ID: MomentAI_4665", self)
        self.label2_1.setAlignment(Qt.AlignCenter)
        
        #self.label2_2 = QLabel("ID: MomentAI_4665", self)
        #self.label2_2.setAlignment(Qt.AlignCenter)
        
        #self.label2_3 = QLabel(status+person2_status, self)
        #self.label2_3.setAlignment(Qt.AlignCenter)
        
        
        self.progress_bar1 = QProgressBar()
        self.progress_bar1.setRange(0, 100)  # Set the range (0-100)
        #self.progress_bar1.setValue(80)
        self.progress_bar1.setTextVisible(False)
        self.bottom_box1_layout.addWidget(self.progress_bar1)
            
        
        # Add labels to the box layouts
        self.bottom_box1_layout.addWidget(self.label1_1)
        #self.bottom_box1_layout.addWidget(self.label1_2)
        #self.bottom_box1_layout.addWidget(self.label1_3)
        
        
        self.progress_bar2 = QProgressBar()
        self.progress_bar2.setRange(0, 100)  # Set the range (0-100)
        #self.progress_bar2.setValue(80)
        self.progress_bar2.setTextVisible(False)
        self.bottom_box2_layout.addWidget(self.progress_bar2)

        self.bottom_box2_layout.addWidget(self.label2_1)
        #self.bottom_box2_layout.addWidget(self.label2_2)
        #self.bottom_box2_layout.addWidget(self.label2_3)

        # Initialize the camera as None (not started)
        self.capture = None

        # Create a QTimer to update the camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Connect button clicks to functions
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        #self.quit_button.clicked.connect(self.quit_application)

        # Create layouts for the entire UI
        layout = QVBoxLayout()
        #layout.addWidget(self.image_container)
        layout.addWidget(self.label)
        
        shared_image_layout_person1 = QHBoxLayout()
        self.image_labels_person1 = [] 
        image_sources = ['default/sleepy.png','default/yawning.png','default/heartAttack.png','default/epilepsy.png']
        for i in range(4):
            image_label = QLabel(self)
            image = QPixmap(image_sources_default[i])
            image = image.scaled(60, 60)
            image_label.setPixmap(image)
            shared_image_layout_person1.addStretch(1)  # Add spacing before each image
            shared_image_layout_person1.addWidget(image_label)
            shared_image_layout_person1.addStretch(1) 
            self.image_labels_person1.append(image_label)
            
            
        shared_image_layout_person2 = QHBoxLayout()
        self.image_labels_person2 = [] 
        image_sources = ['gw.png','logo.jpg','logo.jpg','gw.png']
        for i in range(4):
            image_label = QLabel(self)
            image = QPixmap(image_sources_default[i])
            image = image.scaled(60, 60)
            image_label.setPixmap(image)
            shared_image_layout_person2.addStretch(1)  # Add spacing before each image
            shared_image_layout_person2.addWidget(image_label)
            shared_image_layout_person2.addStretch(1) 
            self.image_labels_person2.append(image_label)
        
        shared_images_data_layout = QVBoxLayout() 
        # Add the bottom boxes to the layout side by side
        bottom_boxes_layout = QHBoxLayout()
        shared_image_layout = QHBoxLayout()
        self.shared_image_labels = [] 
        image_sources = ['gw.png','logo.jpg','logo.jpg','gw.png']
        for i in range(4):
            image_label = QLabel(self)
            image = QPixmap(image_sources_shared_default[i])
            image = image.scaled(60, 60)
            image_label.setPixmap(image)
            shared_image_layout.addStretch(1)  # Add spacing before each image
            shared_image_layout.addWidget(image_label)
            shared_image_layout.addStretch(1) 
            self.shared_image_labels.append(image_label)
        #bottom_boxes_layout.addSpacing(15)
        self.bottom_box1_layout.addLayout(shared_image_layout_person1);
        bottom_boxes_layout.addWidget(self.bottom_box1)
        gap_widget = QWidget()
        gap_widget.setFixedWidth(15)  # Adjust the gap height as needed
        bottom_boxes_layout.addWidget(gap_widget)
        self.bottom_box2_layout.addLayout(shared_image_layout_person2);
        bottom_boxes_layout.addWidget(self.bottom_box2)
        self.bottom_box1.setVisible(False)
        self.bottom_box2.setVisible(False)
        self.bottom_layout_widget = QWidget()
        shared_images_data_layout.addLayout(shared_image_layout)
        shared_images_data_layout.addLayout(bottom_boxes_layout)
        
        #shared_image_layout.addLayout(bottom_boxes_layout)
        self.bottom_layout_widget.setLayout(shared_images_data_layout)
        #self.bottom_layout_widget.addLayout(bottom_boxes_layout)
        self.bottom_layout_widget.setStyleSheet("background-color: gray;border-radius:15px;") 
        

        
        
        self.bottom_layout_widget.setVisible(False)
        layout.addWidget(self.bottom_layout_widget)
    
        
        layout.addWidget(button_container)
        layout.addLayout(button_layout2)

        container = QWidget()
        #container.setStyleSheet("background-color: white;");
        container.setLayout(layout)
        self.setCentralWidget(container)
        
    def generate_random_numbers(interval_seconds):
        while True:
            random_number = random.randint(1, 100)  # Generate a random number between 1 and 100 (adjust as needed)
            #print(f"Random number: {random_number}")
            time.sleep(interval_seconds)  

    def load_images(self):
        # Load and set images for top left, middle, and top right labels with a size of 60x60 pixels
        left_image = QPixmap("images1/gw.png").scaled(100, 100)
        middle_image = QPixmap("images1/gw_horizontal_2c.png").scaled(280, 60)
        right_image = QPixmap("images1/logo.jpg").scaled(120, 60)

        self.top_left_image.setPixmap(left_image)
        self.middle_image.setPixmap(middle_image)
        self.top_right_image.setPixmap(right_image)

    def start_camera(self):
        if self.capture is None:
            self.bottom_box1.setVisible(True)
            self.bottom_box2.setVisible(True)
            self.bottom_layout_widget.setVisible(True)
            # Initialize the camera
            self.capture = cv2.VideoCapture(1)
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(self.capture.get(3)), int(self.capture.get(4))))
            
            self.timer.start(10)  # Adjust the interval as needed

    def stop_camera(self):
        if self.capture is not None:
            # Release the camera
            self.capture.release()
            self.capture = None
            self.bottom_layout_widget.setVisible(False)
            self.bottom_box1.setVisible(False)
            self.bottom_box2.setVisible(False)
            self.timer.stop()
            self.label.clear()

    def quit_application(self):
        # Close the application
        self.bottom_box1.setVisible(False)
        self.bottom_box2.setVisible(False)
        self.close()
        self.capture.release()
        self.capture = None
    def changeImage(self,i,image_source,label):
        pixmap = QPixmap(image_source).scaled(60, 60)
        if(label == 'person1'):
            self.image_labels_person1[i].setPixmap(pixmap)
        elif(label == 'person2'):
            self.image_labels_person2[i].setPixmap(pixmap)
        elif(label == 'shared'):
            self.shared_image_labels[i].setPixmap(pixmap)
        # Change the image on the QLabel when the button is clicked
        
        
    def set_bar_color(self,bar_number,value):
        # Calculate the gradient color based on the current health value
        #global health
        if 70 <= value <= 80:
            gradient_color = "qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 green, stop:1 red);"
        elif value > 80:
            gradient_color = "qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 red, stop:1 orange);"
        else:
            gradient_color = "qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 lime, stop:1 green);"
        
        # Set the stylesheet with the updated color
        style_sheet = f"QProgressBar::chunk {{ background: {gradient_color} }}"
        
        if(bar_number == 1):
            self.progress_bar1.setStyleSheet(style_sheet)
        else:
            self.progress_bar2.setStyleSheet(style_sheet)
        
    def updateProgressBar(self, bar_number, value):
        print("Updating Progress Bar")
        if bar_number == 1:
            self.progress_bar1.setValue(value)
            
        elif bar_number == 2:
            self.progress_bar2.setValue(value)   
        self.set_bar_color(bar_number,value)
    def update_frame(self):
        global heart_attack_left,heart_attack_right,progressbar1_value,progressbar2_value,image_sources_shared_active,image_sources_shared_default,image_sources_default,image_sources_active,frame_id, left_blink1,left_blink2,left_val1,left_val2,number_of_faces,sleep1, drowsy1, mpoints1, active1, status1, yawn_count1, distraction1, heart_attack1, centroidx1, centroidy1, centroidz1, CA1, CA_signal1, dummyx1, dummyy1, sdvxlist1, sdvylist1, XList1, YList1, XAvgList1, YAvgList1,sleep2, drowsy2, mpoints2, active2, status2, yawn_count2, distraction2, heart_attack2, centroidx2, centroidy2, centroidz2, CA2, CA_signal2, dummyx2, dummyy2, sdvxlist2, sdvylist2, XList2, YList2, XAvgList2, YAvgList2,sleep, drowsy,mpoints, active,status,yawn_count,distraction,heart_attack,centroidx,centroidy,centroidz,CA,CA_signal,dummyx,dummyy,sdvxlist,sdvylist,XList,YList,XAvgList,YAvgList 

        if self.capture is not None:
            ret, frame = self.capture.read()
            

            frame=cv2.flip(frame,-1)
            wid=720
            heigh=480
            
            #frame=cv2.resize(frame, (wid,heigh))
           
            if ret:
                # Convert the OpenCV image to a QImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with FaceMesh
                results = face_mesh.process(frame)
                

                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        # Draw landmarks on the frame
                        mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS, 
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255, 150), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255,150), thickness=1))
                if(results.multi_face_landmarks != None):
                    
                    if(len(results.multi_face_landmarks)==2):
                        self.changeImage(2,image_sources_shared_active[2],'shared')
                        self.changeImage(1,image_sources_shared_default[1],'shared')
                        if len(results.multi_face_landmarks)!=number_of_faces:
                            #print('2222222222222222222222222')
                            max_queue.queue.clear()
                            min_queue.queue.clear()
                            number_of_faces=2
                        mpoints1 =results.multi_face_landmarks[0].landmark
                        mpoints2 =results.multi_face_landmarks[1].landmark
                        
                       
                        z1=mpoints1[93].z*wid  ### considered the depth (### Done)
                        z2=mpoints1[323].z*wid
                        z3 = mpoints1[4].z*wid
                        
                        z1_2=mpoints2[93].z*wid  ### considered the depth (### Done)
                        z2_2=mpoints2[323].z*wid
                        z3_2 = mpoints2[4].z*wid
                        
                        x1=mpoints1[93].x*wid
                        x2=mpoints1[323].x*wid
                        x3=mpoints1[4].x*wid
                        y1=mpoints1[93].y*heigh
                        y2=mpoints1[323].y*heigh
                        y3=mpoints1[4].y*heigh
                        x4=mpoints1[33].x*wid
                        y4=mpoints1[33].y*heigh
                        x5=mpoints1[160].x*wid
                        y5=mpoints1[160].y*heigh
                        x6=mpoints1[158].x*wid
                        y6=mpoints1[158].y*heigh
                        x7=mpoints1[144].x*wid
                        y7=mpoints1[144].y*heigh
                        x8=mpoints1[153].x*wid
                        y8=mpoints1[153].y*heigh
                        x9=mpoints1[133].x*wid
                        y9=mpoints1[133].y*heigh
                        x10=mpoints1[362].x*wid
                        y10=mpoints1[362].y*heigh
                        x11=mpoints1[385].x*wid
                        y11=mpoints1[385].y*heigh
                        x12=mpoints1[387].x*wid
                        y12=mpoints1[387].y*heigh
                        x13=mpoints1[380].x*wid
                        y13=mpoints1[380].y*heigh
                        x14=mpoints1[373].x*wid
                        y14=mpoints1[373].y*heigh
                        x15=mpoints1[263].x*wid
                        y15=mpoints1[263].y*heigh
                        x16=mpoints1[35].x*wid
                        y16=mpoints1[35].y*heigh
                        x17=mpoints1[16].x*wid
                        y17=mpoints1[16].y*heigh
                        x18=mpoints1[315].x*wid
                        y18=mpoints1[315].y*heigh
                        x19=mpoints1[72].x*wid
                        y19=mpoints1[72].y*heigh
                        x20=mpoints1[11].x*wid
                        y20=mpoints1[11].y*heigh
                        x21=mpoints1[302].x*wid
                        y21=mpoints1[302].y*heigh
                        x22=mpoints1[168].x*wid  #prev78 ## Now the mouth distance is normalised by forhead nose ditance. Why? In angle these two points are visible
                        y22=mpoints1[168].y*heigh  ##prev78
                        x23=mpoints1[19].x*wid ## prev308 
                        # print(x23)
                        y23=mpoints1[19].y*heigh ## prev 308
                        leftUp=(mpoints1[19].x*wid,mpoints1[19].y*heigh)
                        
                        x1_2=mpoints2[93].x*wid
                        x2_2=mpoints2[323].x*wid
                        x3_2=mpoints2[4].x*wid
                        y1_2=mpoints2[93].y*heigh
                        y2_2=mpoints2[323].y*heigh
                        y3_2=mpoints2[4].y*heigh
                        x4_2=mpoints2[33].x*wid
                        y4_2=mpoints2[33].y*heigh
                        x5_2=mpoints2[160].x*wid
                        y5_2=mpoints2[160].y*heigh
                        x6_2=mpoints2[158].x*wid
                        y6_2=mpoints2[158].y*heigh
                        x7_2=mpoints2[144].x*wid
                        y7_2=mpoints2[144].y*heigh
                        x8_2=mpoints2[153].x*wid
                        y8_2=mpoints2[153].y*heigh
                        x9_2=mpoints2[133].x*wid
                        y9_2=mpoints2[133].y*heigh
                        x10_2=mpoints2[362].x*wid
                        y10_2=mpoints2[362].y*heigh
                        x11_2=mpoints2[385].x*wid
                        y11_2=mpoints2[385].y*heigh
                        x12_2=mpoints2[387].x*wid
                        y12_2=mpoints2[387].y*heigh
                        x13_2=mpoints2[380].x*wid
                        y13_2=mpoints2[380].y*heigh
                        x14_2=mpoints2[373].x*wid
                        y14_2=mpoints2[373].y*heigh
                        x15_2=mpoints2[263].x*wid
                        y15_2=mpoints2[263].y*heigh
                        x16_2=mpoints2[35].x*wid
                        y16_2=mpoints2[35].y*heigh
                        x17_2=mpoints2[16].x*wid
                        y17_2=mpoints2[16].y*heigh
                        x18_2=mpoints2[315].x*wid
                        y18_2=mpoints2[315].y*heigh
                        x19_2=mpoints2[72].x*wid
                        y19_2=mpoints2[72].y*heigh
                        x20_2=mpoints2[11].x*wid
                        y20_2=mpoints2[11].y*heigh
                        x21_2=mpoints2[302].x*wid
                        y21_2=mpoints2[302].y*heigh
                        x22_2=mpoints2[168].x*wid  #prev78 ## Now the mouth distance is normalised by forhead nose ditance. Why? In angle these two points are visible
                        y22_2=mpoints2[168].y*heigh  ##prev78
                        x23_2=mpoints2[19].x*wid ## prev308 
                        #print(x23_2)
                        midpoint=(x23+x23_2)/2
                        y23_2=mpoints2[19].y*heigh ## prev 308
                        leftUp_2=(mpoints2[19].x*wid,mpoints2[19].y*heigh)
                        number_of_faces=2
                        ####Person1
                        left_val1 = blinked((x4,y4),(x5,y5), 
                            (x6,y6), (x7,y7), (x8,y8), (x9,y9))
                        # Add the left ratio to both queues
                        max_queue1.put(left_val1)  # Use negative values for max priority queue
                        '''
                        if(left_val1 < manualThreshold):
                            min_queue1.put(-left_val1)
                        '''
 
                        ###############################
                        leftframe=frame[:,:int(midpoint)]
                        rightframe=frame[:,int(midpoint):]   
                        cv2.imwrite(f'body_with_surroundings.jpg', leftframe)
                        cv2.imwrite(f'body_with_surroundings2.jpg', rightframe)
                        cropped_object_list = []
                        cropped_object_list.append(leftframe)
                        cropped_object_list.append(rightframe)                                

                            
                        results1 = holistic.process(leftframe)
                        if results1.pose_landmarks!=None:
                            mp_drawing.draw_landmarks(leftframe, results1.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
                        
                            points1 = results1.pose_landmarks.landmark
                            Lsx1= points1[11].x*wid/2 ##11: Left shoulder 20: Right index finger
                            Lsy1=points1[11].y*heigh
                            
                            Rix1= points1[20].x*wid/2 
                            Riy1=points1[20].y*heigh
                            
                            HA1 = compute((Lsx1,Lsy1),(Rix1,Riy1))
                            #print("HA1= " +str(HA1))
    
                            if HA1 <150 :
                                heart_attack1+=1
                                
                                i=0
                                if(heart_attack1>8):
                                    print("heart_attack1 = " + str( heart_attack1))
                                    i+=1
                                    
                                    #cv2.putText(image, "Heart Attack", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                    #cv2.imshow('MediaPipe FaceMesh', image)
                                    
                                    #if i==1:
                                        #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
                            else:
                              heart_attack1=0
                              #self.label1_3.setText("")
                        heart_attack_left=heart_attack1

                        cv2.imshow("left", leftframe)
                        results2 = holistic.process(rightframe)
                        if results2.pose_landmarks!=None:
                            
                            mp_drawing.draw_landmarks(rightframe, results2.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
                    
                            points2 = results2.pose_landmarks.landmark
                            

                            
                            Lsx2= points2[11].x*wid/2  ##11: Left shoulder 20: Right index finger
                            Lsy2=points2[11].y*heigh
                            
                            Rix2= points2[20].x*wid/2
                            Riy2=points2[20].y*heigh
                            
                            HA2 = compute((Lsx2,Lsy2),(Rix2,Riy2))
                            #print("HA2=  " + str(HA2))
                        
                            if HA2 <150 :
                                heart_attack2+=1
                                
                                i=0
                                if(heart_attack2>8):
                                    print("heart_attack2 = " + str( heart_attack2))
                                    i+=1
                                    #self.label2_3.setText("Heart Attack");
                                    #cv2.putText(image, "Heart Attack", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                    #cv2.imshow('MediaPipe FaceMesh', image)
                                    
                                    #if i==1:
                                        #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
                            else:
                              heart_attack2=0 
                              #self.label2_3.setText("")
                        heart_attack_right=heart_attack2
                        cv2.imshow("right", rightframe)
                        #################################
                        
                        min_queue1.put(-left_val1)
                        # If the queues exceed their maximum size, remove the highest/lowest element
                        if max_queue1.qsize() >qsize :
                            max_queue1.get()
                        if min_queue1.qsize() > qsize:
                            min_queue1.get()        

                        right_val1 = blinked((x10,y10),(x11,y11), 
                            (x12,y12), (x13,y13), (x14,y14), (x15,y15))


                        # If the queues exceed their maximum size, remove the highest/lowest element
                        if max_queue1.qsize() >qsize :
                            max_queue1.get()
                        if min_queue1.qsize() > qsize:
                            min_queue1.get()
                        min_queuee1=[]
                        for e in min_queue1.queue:
                            min_queuee1.append(-e)
                       # print(max_queue1.queue)
                       # print(min_queuee1)  
                        
                        max_average1 = sum(list(max_queue1.queue)) / max_queue1.qsize() if max_queue1.qsize() > 0 else 0
                        if(len(min_queue1.queue)>0):
                            min_average1 = -sum(list(min_queue1.queue)) / min_queue1.qsize() if min_queue1.qsize() > 0 else 0
                            threshhold1=min_average1+0.5*(max_average1-min_average1)
                            #threshhold1=min(max_queue.queue)
                            
                            #threshhold=min(threshhold,manualThreshold)
                            
                            print('left value1  ' +str(left_val1))
                            #print('right value  '+str(right_val))
                            print('threshhold1  '+str(threshhold1))
                            left_blink1=detect(left_val1, threshhold1)
                            right_blink1=detect(left_val1, threshhold1)





                        yawn_g1 = yawn((x16,y16),(x17,y17),(x18,y18),(x19,y19),(x20,y20),(x21,y21),(x22,y22),(x23,y23))
                    
                        #####Person2
                        left_val2 = blinked((x4_2,y4_2),(x5_2,y5_2), 
                            (x6_2,y6_2), (x7_2,y7_2), (x8_2,y8_2), (x9_2,y9_2))
                        # Add the left ratio to both queues
                        max_queue2.put(left_val2)  # Use negative values for max priority queue
                        '''
                        if(left_val2 < manualThreshold):
                            min_queue2.put(-left_val2)
                        '''
                        min_queue2.put(-left_val2)
                        
                        # If the queues exceed their maximum size, remove the highest/lowest element
                        if max_queue2.qsize() >qsize :
                            max_queue2.get()
                        if min_queue2.qsize() > qsize:
                            min_queue2.get()        
            
            
            
            
            
    
                        right_val2 = blinked((x10_2,y10_2),(x11_2,y11_2), 
                            (x12_2,y12_2), (x13_2,y13_2), (x14_2,y14_2), (x15_2,y15_2))
    

                        # If the queues exceed their maximum size, remove the highest/lowest element
                        if max_queue2.qsize() >qsize :
                            max_queue2.get()
                        if min_queue2.qsize() > qsize:
                            min_queue2.get()
                        min_queuee2=[]
                        for e in min_queue2.queue:
                            min_queuee2.append(-e)
                        #print(max_queue2.queue)
                        #print(min_queuee2)  
                        
                        max_average2 = sum(list(max_queue2.queue)) / max_queue2.qsize() if max_queue2.qsize() > 0 else 0
                        if(len(min_queue2.queue)>0):
                            min_average2 = -sum(list(min_queue2.queue)) / min_queue2.qsize() if min_queue2.qsize() > 0 else 0
                            threshhold2=min_average2+0.5*(max_average2-min_average2)
                            #threshhold1=min(max_queue.queue)
                            
                            #threshhold=min(threshhold,manualThreshold)
                            
                            print('left value2  ' +str(left_val2))
                            #print('right value  '+str(right_val))
                            print('threshhold2  '+str(threshhold2))
                            left_blink2=detect(left_val2, threshhold2)
                            right_blink2=detect(left_val2, threshhold2)






                            
                        yawn_g2 = yawn((x16_2,y16_2),(x17_2,y17_2),(x18_2,y18_2),(x19_2,y19_2),(x20_2,y20_2),(x21_2,y21_2),(x22_2,y22_2),(x23_2,y23_2))    
                    
                    
                    
                    
                    ###person1
                    
                        if (yawn_g1 ==3):
                           yawn_count1 +=1
                           #print(yawn_count)
                           if(yawn_count1>15):
                                   print('yawning121212121')
                                   #self.label1_3.setText("Yawning");
                                   #cv2.putText(image, "person1 Yawning", (500,800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),4)
                                   #cv2.imshow('MediaPipe FaceMesh', image)
                                   #if (yawn_count==16):
                                      #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan  is yawning while driving")
                      
                        else:
                           yawn_count1=0    
                           #self.label1_3.setText("");
                     
                    ###person2
                        
                        if (yawn_g2 ==3):
                           yawn_count2 +=1
                           #print(yawn_count)
                           if(yawn_count2 >15):
                                   #Qfdef 
                                   print('yawning')
                                   #self.label2_3.setText("Yawning");
                                   #cv2.putText(image, "person2 Yawning", (500,800), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),4)
                                   #cv2.imshow('MediaPipe FaceMesh', image)
                                   #if (yawn_count==16):
                                      #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan  is yawning while driving")
                      
                        else:
                           yawn_count2 =0   
                           #self.label2_3.setText("");
                    
                    
                    
                    
                    
                    #Person1
                        if(left_blink1==0 or right_blink1==0):
                          sleep1 +=1
                          drowsy1 =0
                          active1 =0
                          #print('sleeep1 '+str(sleep1))
                          if(sleep1 >20):
                               
                                print("PERSON 1 IS SLEEPING !!!")
                                #self.label1_3.setText("Sleeping");
        
                                #if i==1:
                                   #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan  slept while driving")
                                #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan  slept while driving")
                    # =============================================================================
                        elif(left_blink1 ==1 or right_blink1 ==1):
                           sleep1=0
                    #       active1 =0
                    #       drowsy1 +=1
                    #       if(drowsy1 >48):
                    #             status1 ="Drowsy !"
                    #             #print(status)
                    #             color = (0,0,255)
                    #             #cv2.putText(image, status, (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                    #             cv2.imshow('MediaPipe FaceMesh', image)
                    #             #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was drowsy while driving")
                        else:
                    #       drowsy1 =0
                           sleep1 =0
                    #       active1 +=1
                    #       if(active1 >6):
                    #             status1 ="Active :)"
                    #             color = (0,255,0)
                    #             #cv2.putText(image, status, (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                    #   
                    # =============================================================================
                    
                    ## person2
                    
                        if(left_blink2 ==0 or right_blink2 ==0):
                          sleep2 +=1
                          drowsy2 =0
                          active2 =0
                          #print('sleep2  ' + str(sleep2))
                          if(sleep2 >20):
                                #print(" PERSON 2 IS SLEEPING !!!")
                                #self.label2_3.setText("Sleeping");
                                print(status)
                        elif(left_blink2 ==1 or right_blink2 ==1):
                           sleep2 =0
                    #       active1 =0
                    #       drowsy1 +=1
                    #       if(drowsy1 >48):
                    #             status1 ="Drowsy !"
                    #             #print(status)
                    #             color = (0,0,255)
                    #             #cv2.putText(image, status, (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                    #             cv2.imshow('MediaPipe FaceMesh', image)
                    #             #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was drowsy while driving")
                        else:
                    #       drowsy1 =0
                           sleep2 =0
                     ################################################## seizure for two person
                    
                     #### 1st person
                        frames_observed=20
                        centroid1 = [(x1+x2+x3)/3, (y1+y2+y3)/3, (z1+z2+z3)/3]
                        centroid1=[mpoints1[19].x*wid,mpoints1[19].y*heigh,mpoints1[19].z*wid]
                        centroidx1.append(centroid1[0])
                        centroidy1.append(centroid1[1])
                        centroidz1.append(centroid1[2])
                        if len(centroidx1) > frames_observed:
                           centroidx1.pop(0)
                        sdx1 = st.stdev(centroidx1)
                               
                    
                        if len(centroidy1) > frames_observed:
                           centroidy1.pop(0)
                        sdy1 = st.stdev(centroidy1)
                        sdvxlist1.append(sdx1)
                        sdvylist1.append(sdy1)
                       
                       
                        if len(sdvxlist1)>frames_observed: #30 here reprents CA threshhold
                           sdvxlist1.pop(0)
                        if len(sdvylist1)>frames_observed: #30 here reprents CA threshhold
                           sdvylist1.pop(0)
                       
                        if len(centroidz1) > frames_observed:
                           centroidz1.pop(0)
                        sdz1 = st.stdev(centroidz1)
                      
######################## wisso's speed work:
                        if len(centroidx1)>2:
                            vx,vy,vz,v_magnitude   =velocities(centroidx1,centroidy1,centroidz1,frames_observed)       
                            signsx=[]
                            for i,e in enumerate(vx):

                                if i==0:
                                    signsx.append(e)
                                elif signsx[-1]*e<0 and abs(e)>400:
                                    signsx.append(e)
                            #print('number of sign changes 1: '+ str(len(signsx)-1))
                            #print("sign sx " + str(signsx))
    
                            signsy=[]
                            for i,e in enumerate(vy):

                                if i==0:
                                    signsy.append(e)
                                elif signsy[-1]*e<0 and abs(e)>400:
                                    signsy.append(e)
                            #print('number of sign changes 2: '+ str(len(signsy)-1))
                            #print("sign sy " + str(signsy))
                            CA1=max(len(signsx),len(signsy))
                            if len(signsx)>=4 or len(signsy)>=4:
                                print('EXOGENEOUS HEALTH EVENT 111111111111111111111111111')
                                
    
    
    
    
    
    
    
    
                            ax,ay,az,a_magnitude= acceleration(vx,vy,vz,frames_observed)
    
                            jx,jy,jz,j_magnitude= jerk(ax,ay,az,frames_observed)
                            #print('jerk 1 is '+ str(a_magnitude))
                            
                            
                            
                            
# =============================================================================
#                             diff_ax=abs(ax[-1]-ax[0])
#                             diff_ay=abs(ay[-1]-ay[0])  
#                             diff_az=abs(az[-1]-az[0]) 
#                             print('difference in ax is '+ str(diff_ax))
#                             print('difference in ay is '+ str(diff_ay))
#                             print('difference in az is '+ str(diff_az))
#                             print('acceleration is ' +str(a_magnitude))
# =============================================================================
                            
#########################
                        avgsdx1=sum(sdvxlist1[-15:])/len(sdvxlist1[-15:])
                        avgsdy1=sum(sdvylist1[-15:])/len(sdvylist1[-15:])
                        #if sum(sdvxlist[-30:])>sum(sdvxlist[0:30])or sum(sdvylist[-30:])>sum(sdvylist[0:30]):
                        if sdx1>avgsdx1 or sdy1>avgsdy1:
                          increasing=True
                        else:
                          increasing=False
                        if c1%20==0:
                          dummyx1=sdx1
                          dummyy1=sdy1       
                        #print("sdx1 = " + str(sdx1) + " sdy1 = "+ str(sdy1)+ " sdz1="+str(sdz1))
                    
# =============================================================================
#                                
#                         if (sdx1>5 or sdy1 > 5 or sdz1>8)and increasing==True :
#                            sleep1=0
#                            active1=0
#                            
#                            
#                            #print(increasing)
#                            CA1+=1
#                            #print("CA1 is= "+str(CA1))
#                            #print(CA)
#                            i=0
#                            if(CA1>30):
#                              
#                               distraction1=0  
#                               i+=1
#                               
#                               #print('exogeneous health event 1')
#                               #self.label1_3.setText("exogeneous health event");
#                               #cv2.putText(image, "Exogeneous health event", (400,300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),3)
#                               #if CA==111:
#                                  #message = client.messages.create(to='+3028267898',from_="+19705358298",body=" Megan is having exogeneous events while driving")
#                               #cv2.imshow('MediaPipe FaceMesh', image)
#                               #cv2.waitKey(1)
#                         else:
#                            CA1=0
# =============================================================================
                        ########### person 2   
                        centroid2 = [(x1_2+x2_2+x3_2)/3, (y1_2+y2_2+y3_2)/3, (z1_2+z2_2+z3_2)/3]
                        centroidx2.append(centroid2[0])
                        centroidy2.append(centroid2[1])
                        centroidz2.append(centroid2[2])
                               
                        if len(centroidx2) > frames_observed:
                           centroidx2.pop(0)
                        sdx2 = st.stdev(centroidx2)
                               
                    
                        if len(centroidy2) > frames_observed:
                           centroidy2.pop(0)
                        
                        if len(centroidz2) > frames_observed:
                           centroidz2.pop(0)  
                        
                        sdy2 = st.stdev(centroidy2)
                        sdvxlist2.append(sdx2)
                        sdvylist2.append(sdy2)
                       
                        if len(centroidx2)>2:
                            vx,vy,vz,v_magnitude   =velocities(centroidx2,centroidy2,centroidz2,frames_observed)       
                            signsx=[]
                            for i,e in enumerate(vx):

                                if i==0:
                                    signsx.append(e)
                                elif signsx[-1]*e<0 and abs(e)>400:
                                    signsx.append(e)
                            print('number of sign changes 1: '+ str(len(signsx)-1))
                            print("sign sx " + str(signsx))
    
                            signsy=[]
                            for i,e in enumerate(vy):

                                if i==0:
                                    signsy.append(e)
                                elif signsy[-1]*e<0 and abs(e)>400:
                                    signsy.append(e)
                            #print('number of sign changes 2: '+ str(len(signsy)-1))
                            #print("sign sy " + str(signsy))
                            CA2=max(len(signsx),len(signsy))
                            if len(signsx)>=4 or len(signsy)>=4:
                                
                                print('EXOGENEOUS HEALTH EVENT 222222222222222222222222222222222222222222222')
                            
                            
                            ax,ay,az,a_magnitude= acceleration(vx,vy,vz,frames_observed)
                            
                            jx,jy,jz,j_magnitude= jerk(ax,ay,az,frames_observed)
                            print('jerk 2 is '+ str(a_magnitude))
                       
                        
                       
                        if len(sdvxlist2)>30: #30 here reprents CA threshhold
                           sdvxlist2.pop(0)
                        if len(sdvylist2)>30: #30 here reprents CA threshhold
                           sdvylist2.pop(0)
                       
                        if len(centroidz2) > 30:
                           centroidz2.pop(0)
                        sdz2 = st.stdev(centroidz2)
                       
                    
                        avgsdx2=sum(sdvxlist2[-15:])/len(sdvxlist2[-15:])
                        avgsdy2=sum(sdvylist2[-15:])/len(sdvylist2[-15:])
                        #if sum(sdvxlist[-30:])>sum(sdvxlist[0:30])or sum(sdvylist[-30:])>sum(sdvylist[0:30]):
                        if sdx2>avgsdx2 or sdy2>avgsdy2:
                          increasing=True
                        else:
                          increasing=False
                        if c2%20==0:
                          dummyx2=sdx2
                          dummyy2=sdy2      
                        #print("sdx2 = " + str(sdx2) + " sdy2 = "+ str(sdy2)+ " sdz2="+str(sdz2))
                    
# =============================================================================
#                                
#                         if (sdx2>5 or sdy2 > 5 or sdz2>8)and increasing==True :
#                            sleep2=0
#                            active2=0
#                            
#                            
#                            #print(increasing)
#                            CA2+=1
#                            #print("CA2 is "+ str(CA2))
#                          
#                            i=0
#                            if(CA2>30):
#                              
#                               distraction2=0  
#                               i+=1
#                               
#                               #print('exogeneous health event 2')
#                               #self.label2_3.setText("exogeneous health event");
#                               #cv2.putText(image, "Exogeneous health event", (400,300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),3)
#                               #if CA==111:
#                                  #message = client.messages.create(to='+3028267898',from_="+19705358298",body=" Megan is having exogeneous events while driving")
#                               #cv2.imshow('MediaPipe FaceMesh', image)
#                               #cv2.waitKey(1)
#                         else:
#                            CA2=0                           
#                                    
# =============================================================================
                     
                     
                     
                     
                     ####################################################
  
                    elif(len(results.multi_face_landmarks)==1):
                        self.changeImage(1,image_sources_shared_active[1],'shared')
                        self.changeImage(2,image_sources_shared_default[2],'shared')
                        if len(results.multi_face_landmarks)!=number_of_faces:
                            #print('1'*50)
                            max_queue.queue.clear()
                            min_queue.queue.clear()
                            number_of_faces=1
                        mpoints1 =results.multi_face_landmarks[0].landmark
                        mpoints2 = mpoints1
                        mpoints=results.multi_face_landmarks[0].landmark
                        left_blink=0
                        right_blink=0
                        
               ###################################################         
                        results1 = holistic.process(frame)
                        if results1.pose_landmarks!=None:
                        
                            points1 = results1.pose_landmarks.landmark
                            Lsx1= points1[11].x*wid/2 ##11: Left shoulder 20: Right index finger
                            Lsy1=points1[11].y*heigh
                            
                            Rix1= points1[20].x*wid/2 
                            Riy1=points1[20].y*heigh
                            
                            HA1 = compute((Lsx1,Lsy1),(Rix1,Riy1))
                            print("HA1= " +str(HA1))
                       
                            if HA1 <150 :
                                heart_attack1+=1
                                
                                i=0
                                if(heart_attack1>8):
                                    print("heart_attack1 = " + str( heart_attack1))
                                    i+=1
                                    #self.label1_3.setText("Heart Attack");
                                    #cv2.putText(image, "Heart Attack", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                    #cv2.imshow('MediaPipe FaceMesh', image)
                                    
                                    #if i==1:
                                        #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
                            else:
                              heart_attack1=0
                              #self.label1_3.setText("")

        #######################################################
        
# =============================================================================
#                         results1 = holistic.process(frame)
#                         
#                         left_blink=0
#                         right_blink=0
#                         if results1.pose_landmarks:
#                             # Iterate through each pose landmark using index numbers
#                             for idx, landmark in enumerate(results1.pose_landmarks.landmark):
#                                 mp_drawing.draw_landmarks(frame, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#                                 x = landmark.x
#                                 y = landmark.y
#                                 z = landmark.z
#                                 
#                         if results1.pose_landmarks is not None:        
#                             points= results1.pose_landmarks.landmark
#                         
#                     # =============================================================================
#                     #     11: Left shoulder
#                     #     20: Right index finger
#                     # =============================================================================
#                         
#                             Lsx= points[11].x*wid 
#                             Lsy=points[11].y*heigh
#                             
#                             Rix= points[20].x*wid 
#                             Riy=points[20].y*heigh
#                             
#                             HA = compute((Lsx,Lsy),(Rix,Riy))
#                         
#                             if HA <50 :
#                                 sleep=0
#                                 active=0
#                                 CA=0
#                                 distraction=0
#                                 heart_attack+=1
#                                 
#                                 i=0
#                                 if(heart_attack>20):
#                                     print("heart_attack = " + str( heart_attack))
#                                     i+=1
#                                     self.label1_3.setText("Heart Attack");
#                                     #cv2.putText(image, "Heart Attack", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
#                                     #cv2.imshow('MediaPipe FaceMesh', image)
#                                     
#                                     #if i==1:
#                                         #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
#                             else:
#                               heart_attack=0      
#                             
#                             print(HA)
# =============================================================================
                            
                        
                        
                        
                        
                        #cv2.imshow('MediaPipe FaceMesh', image)
                        #cv2.waitKey(1)
                        #mpoints=results.multi_face_landmarks[0].landmark
                    
                    
                        a1x= mpoints[145].x*wid 
                        a1y=mpoints[145].y*heigh
                        a2x= mpoints[374].x*wid 
                        a2y=mpoints[374].y*heigh
                    
                        w= compute((a1x,a1y),(a2x,a2y))
                        W= 6.3
                        #d= 40
                        ##f = w*d/W
                        #print(f)
                        f= 655
                        d= (W*f)/w 
                        
                        d=(d/0.082724)
                        
                        #1 pixel = 0.0264583333 cm actually
                        # for mediapipe 1pixel = 0.082724 cm
                    
                        x1=mpoints[93].x*wid
                        x2=mpoints[323].x*wid
                        x3=mpoints[4].x*wid
                        y1=mpoints[93].y*heigh
                        y2=mpoints[323].y*heigh
                        y3=mpoints[4].y*heigh
                        
                        
                        z1=mpoints[93].z*wid  ### considered the depth (### Done)
                        z2=mpoints[323].z*wid
                        z3 = mpoints[4].z*wid
                        
                        #z3= -1/mpoints[4].z*d
                        
                        #print(z1) 
                        
                    
                        x4=mpoints[33].x*wid
                        y4=mpoints[33].y*heigh
                        x5=mpoints[160].x*wid
                        y5=mpoints[160].y*heigh
                        x6=mpoints[158].x*wid
                        y6=mpoints[158].y*heigh
                        x7=mpoints[144].x*wid
                        y7=mpoints[144].y*heigh
                        x8=mpoints[153].x*wid
                        y8=mpoints[153].y*heigh
                        x9=mpoints[133].x*wid
                        y9=mpoints[133].y*heigh
                        
                        
                        x10=mpoints[362].x*wid
                        y10=mpoints[362].y*heigh
                        x11=mpoints[385].x*wid
                        y11=mpoints[385].y*heigh
                        x12=mpoints[387].x*wid
                        y12=mpoints[387].y*heigh
                        x13=mpoints[380].x*wid
                        y13=mpoints[380].y*heigh
                        x14=mpoints[373].x*wid
                        y14=mpoints[373].y*heigh
                        x15=mpoints[263].x*wid
                        y15=mpoints[263].y*heigh
                        
                        
                        
                        x16=mpoints[35].x*wid
                        y16=mpoints[35].y*heigh
                        x17=mpoints[16].x*wid
                        y17=mpoints[16].y*heigh
                        x18=mpoints[315].x*wid
                        y18=mpoints[315].y*heigh
                        x19=mpoints[72].x*wid
                        y19=mpoints[72].y*heigh
                        x20=mpoints[11].x*wid
                        y20=mpoints[11].y*heigh
                        x21=mpoints[302].x*wid
                        y21=mpoints[302].y*heigh
                        x22=mpoints[168].x*wid  #prev78 ## Now the mouth distance is normalised by forhead nose ditance. Why? In angle these two points are visible
                        y22=mpoints[168].y*heigh  ##prev78
                        x23=mpoints[19].x*wid ## prev308 
                        y23=mpoints[19].y*heigh ## prev 308
                     
                        leftUp=(mpoints[19].x*wid,mpoints[19].y*heigh)
                    
                    
                    
                    
                    
                    
                        left_val = blinked((x4,y4),(x5,y5), 
                            (x6,y6), (x7,y7), (x8,y8), (x9,y9))
                        
                        max_queue.put(left_val)  # Use negative values for max priority queue
                        '''
                        if(left_val < manualThreshold):
                            min_queue.put(-left_val)
                        '''
                        min_queue.put(-left_val)
                        # If the queues exceed their maximum size, remove the highest/lowest element
                        if max_queue.qsize() >qsize :
                            max_queue.get()
                        if min_queue.qsize() > qsize:
                            min_queue.get()        
                        right_val = blinked((x10,y10),(x11,y11), 
                            (x12,y12), (x13,y13), (x14,y14), (x15,y15))
                        #max_queue.put(right_val)  # Use negative values for max priority queue
                        #min_queue.put(-right_val)

                        # If the queues exceed their maximum size, remove the highest/lowest element
                        if max_queue.qsize() >qsize :
                            max_queue.get()
                        if min_queue.qsize() > qsize:
                            min_queue.get()
                        min_queuee=[]
                        for e in min_queue.queue:
                            min_queuee.append(-e)
                        #print('max')
                        #print(max_queue.queue)
                        #print('min')
                        #print(min_queuee)  
                        
                        max_average = sum(list(max_queue.queue)) / max_queue.qsize() if max_queue.qsize() > 0 else 0
                        if(len(min_queue.queue)>0):
                            min_average = -sum(list(min_queue.queue)) / min_queue.qsize() if min_queue.qsize() > 0 else 0
                            threshhold=min_average+0.6*(max_average-min_average)
                            #threshhold1=min(max_queue.queue)
                            
                            #threshhold=min(threshhold,manualThreshold)
                            
                            #print('left value  ' +str(left_val))
                            #print('right value  '+str(right_val))
                            #print('threshhold  '+str(threshhold))
                            left_blink=detect(left_val, threshhold)
                            right_blink=detect(left_val, threshhold)
                                
                        yawn_g = yawn((x16,y16),(x17,y17),(x18,y18),(x19,y19),(x20,y20),(x21,y21),(x22,y22),(x23,y23))
                    
                    
                    
                    
                    
                    
                    
                        if (yawn_g ==3):
                           yawn_count+=1
                           #print(yawn_count)
                           if(yawn_count>15):
                                   print('yawning')
                                   #self.label1_3.setText("Yawning");
                                   #cv2.putText(image, "Yawning", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                   #cv2.imshow('MediaPipe FaceMesh', image)
                                   #if (yawn_count==16):
                                      #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan  is yawning while driving")
                      
                        else:
                           yawn_count=0   
                           #self.label1_3.setText("");
                                        #Now judge what to do for the eye blinks
                        if(left_blink==0 or right_blink==0):
                          sleep+=1
                          drowsy=0
                          active=0
                          i=0
                          if(sleep>48):
                                i+=1
                                status="SLEEPING !!!"
                                #self.label1_3.setText("Sleeping");
                                
                                #print(status)
                                color = (255,0,0)
                                #cv2.putText(image, status, (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                #cv2.imshow('MediaPipe FaceMesh', image)
                                #if i==1:
                                   #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan  slept while driving")
                                #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan  slept while driving")
                        elif(left_blink==1 or right_blink==1):
                          sleep=0
                          active=0
                          drowsy+=1
                          if(drowsy>48):
                                status="Drowsy !"
                                #print(status)
                                color = (0,0,255)
                                #cv2.putText(image, status, (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                #cv2.imshow('MediaPipe FaceMesh', image)
                                #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was drowsy while driving")
                        else:
                          drowsy=0
                          sleep=0
                          active+=1
                          if(active>6):
                                status="Active :)"
                                color = (0,255,0)
                                #cv2.putText(image, status, (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                    
                                
                           
                                
                    # =============================================================================
                    #     dis = compute(leftUp, (0,0))
                    #     
                    #     print(dis)
                    #     
                    #     if (dis < 250):
                    #         distraction+=1
                    #         print(distraction)
                    # =============================================================================
                            
                            
                        
                        
                                
# =============================================================================
#                         XList.append(leftUp[0])
#                         if len(XList) > 48:
#                             XList.pop(0)
#                         XAvg = ((sum(XList[0:24])/24) - (sum(XList[24:])/24))/2
#                         
#                         #print(abs(XAvg))
#                                 
#                     
#                         YList.append(leftUp[1])
#                         if len(YList) > 48:
#                             YList.pop(0)
#                         YAvg = ((sum(YList[0:24])/24) - (sum(YList[24:])/24))/2
#                                 
#                                 
#                         if (abs(XAvg)> 10 or abs(YAvg) > 10):
#                             sleep=0
#                             active=0
#                             CA=0
#                             distraction+=1
#                             #print(distraction)
#                             i=0
#                             if(distraction>20):
#                                 i+=1
#                                 #cv2.putText(image, "Distracted", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
#                                 #cv2.imshow('MediaPipe FaceMesh', image)
#                                 
#                     
#                                 #if i==1:
#                                     #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
#                         
#                         else:
#                           distraction=0          
# =============================================================================
                                
                    
                                
                    # =============================================================================
                    #     XList.append(leftUp[0])
                    #     if len(XList) > 48:
                    #         XList.pop(0)
                    #     XAvg = ((sum(XList[0:24])/24) - (sum(XList[24:])/24))/2
                    #     
                    #     print(abs(XAvg))
                    #             
                    # 
                    #     YList.append(leftUp[1])
                    #     if len(YList) > 48:
                    #         YList.pop(0)
                    #     YAvg = ((sum(YList[0:24])/24) - (sum(YList[24:])/24))/2
                    #             
                    #             
                    #     if (abs(XAvg)> 10 or abs(YAvg) > 10):
                    #         sleep=0
                    #         active=0
                    #         CA=0
                    #         distraction+=1
                    #         #print(distraction)
                    #         i=0
                    #         if(distraction>200):
                    #             i+=1
                    #             cv2.putText(image, "Distracted", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                    #             cv2.imshow('MediaPipe FaceMesh', image)
                    #             #if i==1:
                    #                 #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
                    #     
                    # 
                    # =============================================================================
                    
                    
                    
                    
                    
                    #### Seizure Start 
                                        
                        centroid = [(x1+x2+x3)/3, (y1+y2+y3)/3, (z1+z2+z3)/3]
                        centroidx.append(centroid[0])
                        centroidy.append(centroid[1])
                        centroidz.append(centroid[2])
                        frames_observed=20

                                 
                        if len(centroidx) > frames_observed:
                           centroidx.pop(0)
                        sdx = st.stdev(centroidx)
                               
                    
                        if len(centroidy) > frames_observed:
                           centroidy.pop(0)
                        sdy = st.stdev(centroidy)
                        sdvxlist.append(sdx)
                        sdvylist.append(sdy)
                       
                       
                        if len(sdvxlist)>30: #30 here reprents CA threshhold
                           sdvxlist.pop(0)
                        if len(sdvylist)>30: #30 here reprents CA threshhold
                           sdvylist.pop(0)
                       
                        if len(centroidz) > frames_observed:
                           centroidz.pop(0)
                        sdz = st.stdev(centroidz)
                        if len(centroidx)>2:
                            vx,vy,vz,v_magnitude   =velocities(centroidx,centroidy,centroidz,frames_observed)       
                            signsx=[]
                            for i,e in enumerate(vx):

                                if i==0:
                                    signsx.append(e)
                                elif signsx[-1]*e<0 and abs(e)>400:
                                    signsx.append(e)
                            #print('number of sign changes 1: '+ str(len(signsx)-1))
                            #print("sign sx " + str(signsx))
    
                            signsy=[]
                            for i,e in enumerate(vy):

                                if i==0:
                                    signsy.append(e)
                                elif signsy[-1]*e<0 and abs(e)>400:
                                    signsy.append(e)
                            #print('number of sign changes 2: '+ str(len(signsy)-1))
                            #print("sign sy " + str(signsy))
                            CA=max(len(signsx),len(signsy))
                            if len(signsx)>=4 or len(signsy)>=4:
                                print('EXOGENEOUS HEALTH EVENT 111111111111111111111111111')
                                
    
    
    
    
    
    
    
    
                            ax,ay,az,a_magnitude= acceleration(vx,vy,vz,frames_observed)
    
                            jx,jy,jz,j_magnitude= jerk(ax,ay,az,frames_observed)
                            #print('jerk 1 is '+ str(a_magnitude))
                            
                    
                        avgsdx=sum(sdvxlist[-15:])/len(sdvxlist[-15:])
                        avgsdy=sum(sdvylist[-15:])/len(sdvylist[-15:])
                        #if sum(sdvxlist[-30:])>sum(sdvxlist[0:30])or sum(sdvylist[-30:])>sum(sdvylist[0:30]):
                        if sdx>avgsdx or sdy>avgsdy:
                          increasing=True
                        else:
                          increasing=False
                        if c%20==0:
                          dummyx=sdx
                          dummyy=sdy        
                        #print("sdx = " + str(sdx) + " sdy = "+ str(sdy)+ " sdz="+str(sdz))
                    
# =============================================================================
#                                
#                         if (sdx>5 or sdy > 5 or sdz>8)and increasing==True :
#                            sleep=0
#                            active=0
#                            
#                            
#                            #print(increasing)
#                            CA+=1
#                            #print(CA)
#                            #print(CA)
#                            i=0
#                            if(CA>30):
#                              
#                               distraction=0  
#                               i+=1
#                               
#                               #print('exogeneous health event')
#                               #self.label1_3.setText("exogeneous health event");
#                               #cv2.putText(image, "Exogeneous health event", (400,300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),3)
#                               #if CA==111:
#                                  #message = client.messages.create(to='+3028267898',from_="+19705358298",body=" Megan is having exogeneous events while driving")
#                               #cv2.imshow('MediaPipe FaceMesh', image)
#                               #cv2.waitKey(1)
#                         else:
#                            CA=0
#                                     
# =============================================================================
                                   
                    else:
                        print("No Person")
                        self.changeImage(1,image_sources_shared_default[1],'shared')
                        self.changeImage(2,image_sources_shared_default[2],'shared')
                        #print('00000000000000000000000000000000000000000000000')
                        max_queue.queue.clear()
                        min_queue.queue.clear()                        
                        number_of_faces=0
                    #image_sources_active = ['active/sleepy.png','active/yawning.png','active/heartAttack.png','active/epilepsy.png']
                    
# =============================================================================
#                     if(CA1-1>0 or CA-1>0):
#                         progressbar1_value = max(CA,CA1)*25
#                         self.updateProgressBar(1, int(progressbar1_value))
# # =============================================================================
# #                     elif(heart_attack >0 or heart_attack1>0):
# #                         progressbar1_value = max(heart_attack,heart_attack1)*12
# #                         self.updateProgressBar(1,int(progressbar1_value))
# # =============================================================================
#                     elif(sleep>=0 or sleep1>0):
#                         progressbar1_value = max(sleep,sleep1)*5
#                         self.updateProgressBar(1, int(progressbar1_value))
#                     elif(yawn_count> 0 or yawn_count1>0):
#                         progressbar1_value = max(yawn_count,yawn_count1)*6
#                         self.updateProgressBar(1, progressbar1_value)
#                     #else:
#                     elif(yawn_count == 0 and yawn_count1 == 0
#                          and sleep == 0 and sleep1 == 0
#                          and heart_attack ==0 and heart_attack1 == 0 
#                          and CA == 1 and CA1 == 1):
#                         if(progressbar1_value <= 5):
#                             progressbar1_value = 0;
#                         else:
#                             progressbar1_value -= 3
#                         self.updateProgressBar(1, progressbar1_value)
# =============================================================================
                    print(" progressbar1_value=  "+ str( progressbar1_value))    
                        
# =============================================================================
#                     if(CA2-1>0):
#                         progressbar2_value = CA2*25
#                         self.updateProgressBar(2, int(progressbar2_value))
#                     elif(heart_attack2>0):
#                         progressbar2_value = heart_attack2*5
#                         self.updateProgressBar(2,int(progressbar2_value))
#                     elif(sleep2>0):
#                         progressbar2_value = sleep2*5
#                         self.updateProgressBar(2, int(progressbar2_value))
#                     elif(yawn_count2>0):
#                         progressbar2_value = yawn_count2*6
#                         self.updateProgressBar(2, progressbar2_value)
#                     elif( yawn_count2 == 0
#                          and sleep2 == 0
#                          and heart_attack2 == 0 
#                          and CA2 == 1):
#                         if(progressbar2_value <= 5):
#                             progressbar2_value = 0;
#                         else:
#                             progressbar2_value -= 3
#                         self.updateProgressBar(2, progressbar2_value)
# =============================================================================
                    print(" progressbar2_value=  "+ str( progressbar2_value))    
                    
                    xc1=mpoints1[19].x
                    xc2=mpoints2[19].x
                    if xc1<xc2:
                        yawn_left=yawn_count1
                        sleep_left=sleep1
                        CA_left=CA1
                        #heart_attack_left=heart_attack1
                        yawn_right=yawn_count2
                        sleep_right=sleep2
                        CA_right=CA2
                        #heart_attack_right=heart_attack2
                    else:
                        yawn_left=yawn_count2
                        sleep_left=sleep2
                        CA_left=CA2
                       # heart_attack_left=heart_attack2
                        
                        yawn_right=yawn_count1
                        sleep_right=sleep1
                        CA_right=CA1
                        #heart_attack_right=heart_attack1
                        
                    

                    health_event=min((CA-1)/4,1)
                    health_event1=min((CA_left-1)/4,1)
                    health_event2=min((CA_right-2)/4,1)
                    
                    sleep_event=min(sleep/20,1)
                    sleep_event1=min(sleep_left/20,1)
                    sleep_event2=min(sleep_right/20,1)
                    
                    yawn_event=min(yawn_count/15,1)
                    yawn_event1=min(yawn_left/15,1)
                    yawn_event2=min(yawn_right/15,1)
                    
                    heart_event=min(heart_attack/8,1)
                    heart_event1=min(heart_attack_left/8,1)
                    heart_event2=min(heart_attack_right/8,1)
                    
                    temp1 = max(health_event,health_event1,sleep_event,sleep_event1,yawn_event,yawn_event1,heart_event,heart_event1);
                    temp2 = max(health_event2,sleep_event2,yawn_event2,heart_event2);
                    temp1 *=100;
                    temp2 *=100;
                    if(temp1<progressbar1_value):
                        progressbar1_value-=5;
                    else:
                        progressbar1_value = temp1;
                    if(temp2<progressbar2_value):
                        progressbar2_value-=5;
                    else:
                        progressbar2_value = temp2;
                    self.updateProgressBar(1, int(progressbar1_value))
                    self.updateProgressBar(2, int(progressbar2_value))
                    
                    
                    if(yawn_left>15 or yawn_count>15 ):
                       # self.label1_3.setText("Yawning");
                        self.changeImage(1,image_sources_active[1],'person1')
                        
                        self.changeImage(0,image_sources_default[0],'person1')
                        self.changeImage(2,image_sources_default[2],'person1')
                        self.changeImage(3,image_sources_default[3],'person1')
                        
                    elif(sleep_left >20 or sleep >20):
                       # self.label1_3.setText("Sleeping");
                        self.changeImage(0,image_sources_active[0],'person1')

                        self.changeImage(1,image_sources_default[1],'person1')
                        self.changeImage(2,image_sources_default[2],'person1')
                        self.changeImage(3,image_sources_default[3],'person1')
                    elif(CA_left>=4 or CA >=4):
                       # self.label1_3.setText("exogeneous health event")
                        self.changeImage(3,image_sources_active[3],'person1')
                        
                        self.changeImage(0,image_sources_default[0],'person1')
                        self.changeImage(1,image_sources_default[1],'person1')
                        self.changeImage(2,image_sources_default[2],'person1')

                    elif(heart_attack_left>8 or heart_attack>8):
                        #self.label1_3.setText("Heart Attack");
                        self.changeImage(2,image_sources_active[2],'person1')
                        
                        self.changeImage(0,image_sources_default[0],'person1')
                        self.changeImage(1,image_sources_default[1],'person1')
                        self.changeImage(3,image_sources_default[3],'person1')
                    else: 
                        #self.label1_3.setText("")
                        self.changeImage(0,image_sources_default[0],'person1')
                        self.changeImage(1,image_sources_default[1],'person1')
                        self.changeImage(2,image_sources_default[2],'person1')
                        self.changeImage(3,image_sources_default[3],'person1')
                    
                    
                    

                    if(yawn_right>15 ):
                        #self.label2_3.setText("Yawning");
                        self.changeImage(1,image_sources_active[1],'person2')
                        
                        self.changeImage(0,image_sources_default[0],'person2')
                        self.changeImage(2,image_sources_default[2],'person2')
                        self.changeImage(3,image_sources_default[3],'person2')
                    elif(sleep_right >20):
                        #self.label2_3.setText("Sleeping");
                        self.changeImage(0,image_sources_active[0],'person2')
                        
                        self.changeImage(1,image_sources_default[1],'person2')
                        self.changeImage(2,image_sources_default[2],'person2')
                        self.changeImage(3,image_sources_default[3],'person2')
                    elif(CA_right>=4):
                        #self.label2_3.setText("exogeneous health event")
                        self.changeImage(3,image_sources_active[3],'person2')
                        
                        self.changeImage(0,image_sources_default[0],'person2')
                        self.changeImage(1,image_sources_default[1],'person2')
                        self.changeImage(2,image_sources_default[2],'person2')
                    elif(heart_attack_right>8):
                        #self.label2_3.setText("Heart Attack");
                        self.changeImage(2,image_sources_active[2],'person2')
                        
                        self.changeImage(0,image_sources_default[0],'person2')
                        self.changeImage(1,image_sources_default[1],'person2')
                        self.changeImage(3,image_sources_default[3],'person2')
                    else:
                        #self.label2_3.setText("")
                        self.changeImage(0,image_sources_default[0],'person2')
                        self.changeImage(1,image_sources_default[1],'person2')
                        self.changeImage(2,image_sources_default[2],'person2')
                        self.changeImage(3,image_sources_default[3],'person2')
                        #if len(results.multi_face_landmarks)!=number_of_faces:

                    
                            
                else:
                    self.changeImage(1,image_sources_shared_default[1],'shared')
                    self.changeImage(2,image_sources_shared_default[2],'shared')
                    #print('00000000000000000000000000000000000000000000000')
                    max_queue.queue.clear()
                    min_queue.queue.clear()                        
                    number_of_faces=0
                       


                dumyf= cv2.resize(frame, (360,240))
                height, width, channel = dumyf.shape
                bytes_per_line = 3 * width
                q_img = QImage(dumyf.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraPreview()
    window.show()
    sys.exit(app.exec_())