# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:13:08 2023

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
status = "Status: "

import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces=5)
#################################

numberOfPerson = 10;

heart_attack1 = 0
heart_attack2 = 0
# Loading camera

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_tracking_confidence=0.5, min_detection_confidence=0.5, model_complexity=1, smooth_landmarks=True)
ww = 480
hh = 480



drowsy = 0
active = 0

yawnCount = [0]*numberOfPerson
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

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()


qsize = 20
num_queues = numberOfPerson

# Create a list of PriorityQueue instances
max_queue = [queue.PriorityQueue(maxsize=qsize+1) for _ in range(num_queues)]
min_queue = [queue.PriorityQueue(maxsize=qsize+1) for _ in range(num_queues)]
max_queue_right = [queue.PriorityQueue(maxsize=qsize+1) for _ in range(num_queues)]
min_queue_right = [queue.PriorityQueue(maxsize=qsize+1) for _ in range(num_queues)]

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
    
    if(yawn_d>1):
        return 3
    else:
        return 0
    
cap = cv2.VideoCapture(0)


numberOfPerson = 10
mPointsList = [0]*numberOfPerson
x1List = [0]*numberOfPerson
y1List = [0]*numberOfPerson
x2List = [0]*numberOfPerson
y2List= [0]*numberOfPerson
x3List= [0]*numberOfPerson
y3List = [0]*numberOfPerson
x4List= [0]*numberOfPerson
y4List= [0]*numberOfPerson
x5List= [0]*numberOfPerson
y5List= [0]*numberOfPerson
x6List= [0]*numberOfPerson
y6List= [0]*numberOfPerson
x7List= [0]*numberOfPerson
y7List= [0]*numberOfPerson
x8List= [0]*numberOfPerson
y8List= [0]*numberOfPerson
x9List= [0]*numberOfPerson
y9List= [0]*numberOfPerson
x10List= [0]*numberOfPerson
y10List= [0]*numberOfPerson
x11List= [0]*numberOfPerson
y11List= [0]*numberOfPerson
x12List= [0]*numberOfPerson
y12List= [0]*numberOfPerson
x13List= [0]*numberOfPerson
y13List= [0]*numberOfPerson
x14List= [0]*numberOfPerson
y14List= [0]*numberOfPerson
x15List= [0]*numberOfPerson
y15List= [0]*numberOfPerson
x16List= [0]*numberOfPerson
y16List= [0]*numberOfPerson
x17List= [0]*numberOfPerson
y17List= [0]*numberOfPerson
x18List= [0]*numberOfPerson
y18List= [0]*numberOfPerson
x19List= [0]*numberOfPerson
y19List= [0]*numberOfPerson
x20List= [0]*numberOfPerson
y20List= [0]*numberOfPerson
x21List= [0]*numberOfPerson
y21List= [0]*numberOfPerson
x22List= [0]*numberOfPerson
y22List= [0]*numberOfPerson
x23List= [0]*numberOfPerson
y23List= [0]*numberOfPerson
yawnGList = [0]*numberOfPerson

left_blink = [0]*numberOfPerson
right_blink = [0]*numberOfPerson
sleep = [0]*numberOfPerson


z1List = [0]*numberOfPerson
z2List = [0]*numberOfPerson
z3List = [0]*numberOfPerson

centroidx = [[0] * numberOfPerson for _ in range(numberOfPerson)]
centroidy = [[0] * numberOfPerson for _ in range(numberOfPerson)]
centroidz = [[0] * numberOfPerson for _ in range(numberOfPerson)]
sdvxlist = [[0] * numberOfPerson for _ in range(numberOfPerson)]
sdvylist = [[0] * numberOfPerson for _ in range(numberOfPerson)]

while cap.isOpened():
    
            ret, frame = cap.read()
            out = frame.copy()
            
            if not ret:
                break
            #frame=cv2.flip(frame,-1)
            wid=720
            heigh=480
            frame=cv2.resize(frame, (wid,heigh))
           
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_mesh.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS, 
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255, 150), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255,150), thickness=1))
                        
                cv2.imshow('MediaPipe FaceMesh', frame)
                if(results.multi_face_landmarks != None):
                    detected_face = len(results.multi_face_landmarks);
                    
                    
                    for person in range(detected_face):                        
                        mPointsList[person] = results.multi_face_landmarks[person].landmark

                        x1List[person] = mPointsList[person][93].x*wid
                        x2List[person] = mPointsList[person][323].x*wid
                        x3List[person] = mPointsList[person][4].x*wid
                        y1List[person] = mPointsList[person][93].y*heigh
                        y2List[person] = mPointsList[person][323].y*heigh
                        y3List[person] = mPointsList[person][4].y*heigh
                        x4List[person] = mPointsList[person][33].x*wid
                        y4List[person] = mPointsList[person][33].y*heigh
                        x5List[person] = mPointsList[person][160].x*wid
                        y5List[person] = mPointsList[person][160].y*heigh
                        x6List[person] = mPointsList[person][158].x*wid
                        y6List[person] = mPointsList[person][158].y*heigh
                        x7List[person] = mPointsList[person][144].x*wid
                        y7List[person] = mPointsList[person][144].y*heigh
                        x8List[person] = mPointsList[person][153].x*wid
                        y8List[person] = mPointsList[person][153].y*heigh
                        x9List[person] = mPointsList[person][133].x*wid
                        y9List[person] = mPointsList[person][133].y*heigh
                        x10List[person] = mPointsList[person][362].x*wid
                        y10List[person] = mPointsList[person][362].y*heigh
                        x11List[person] = mPointsList[person][385].x*wid
                        y12List[person] = mPointsList[person][385].y*heigh
                        x12List[person] = mPointsList[person][387].x*wid
                        y12List[person] = mPointsList[person][387].y*heigh
                        x13List[person] = mPointsList[person][380].x*wid
                        y13List[person] = mPointsList[person][380].y*heigh
                        x14List[person] = mPointsList[person][373].x*wid
                        y14List[person] = mPointsList[person][373].y*heigh
                        x15List[person] = mPointsList[person][263].x*wid
                        y15List[person] = mPointsList[person][263].y*heigh
                        
                        
                        
                        x16List[person] = mPointsList[person][35].x*wid
                        y16List[person] = mPointsList[person][35].y*heigh
                        x17List[person] = mPointsList[person][16].x*wid
                        y17List[person] = mPointsList[person][16].y*heigh
                        x18List[person] = mPointsList[person][315].x*wid
                        y18List[person] = mPointsList[person][315].y*heigh
                        x19List[person] = mPointsList[person][72].x*wid
                        y19List[person] = mPointsList[person][72].y*heigh
                        x20List[person] = mPointsList[person][11].x*wid
                        y20List[person] = mPointsList[person][11].y*heigh
                        x21List[person] = mPointsList[person][302].x*wid
                        y21List[person] = mPointsList[person][302].y*heigh
                        x22List[person] = mPointsList[person][168].x*wid
                        y22List[person] = mPointsList[person][168].y*heigh
                        x23List[person] = mPointsList[person][19].x*wid
                        y23List[person] = mPointsList[person][19].y*heigh
                        
                        yawnGList[person] = yawn((x16List[person],y16List[person]),(x17List[person],y17List[person]),(x18List[person],y18List[person]),(x19List[person],y19List[person]),(x20List[person],y20List[person]),(x21List[person],y21List[person]),(x22List[person],y22List[person]),(x23List[person],y23List[person]))
                        if (yawnGList[person] == 3):
                            yawnCount[person]+=1
                            if(yawnCount[person]>10):
                                    #print('Person'+ str(person+1) +'yawning')
                                    cv2.putText(frame, "Person" + str(person+1) +"Yawning", (100*(person*1),150+(50*person)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                    cv2.imshow('MediaPipe FaceMesh', frame)
                        
                        else:
                            yawnCount[person]=0   
                            
                        
                        
                        ####### Heart Attack ########
# =============================================================================
#                         results1 = holistic.process(frame)
#                         
#                         if results1.pose_landmarks!=None:
#                             print(results1)
#                             #print(len(results1.pose_landmarks))
#                             print("Holistic process", results1.pose_landmarks)
#                         
#                             points1 = results1.pose_landmarks.landmark
#                             Lsx1= points1[11].x*wid/2 ##11: Left shoulder 20: Right index finger
#                             Lsy1=points1[11].y*heigh
#                             
#                             Rix1= points1[20].x*wid/2 
#                             Riy1=points1[20].y*heigh
#                             
#                             HA1 = compute((Lsx1,Lsy1),(Rix1,Riy1))
#                             #print("HA1= " +str(HA1))
#                        
#                             if HA1 <150 :
#                                 heart_attack1+=1
#                                 
#                                 i=0
#                                 if(heart_attack1>8):
#                                     #print("heart_attack1 = " + str( heart_attack1))
#                                     i+=1
#                                     #self.label1_3.setText("Heart Attack");
#                                     cv2.putText(frame, "Heart Attack", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
#                                     cv2.imshow('MediaPipe FaceMesh', frame)
#                                     
#                                     #if i==1:
#                                         #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
#                             else:
#                               heart_attack1=0
# =============================================================================

                        ###########SLeeping################
                        left_val = blinked((x4List[person],y4List[person]),(x5List[person],y5List[person]), 
                          (x6List[person],y6List[person]), (x7List[person],y7List[person]), (x8List[person],y8List[person]), (x9List[person],y9List[person]))
                        right_val = blinked((x10List[person],y10List[person]),(x11List[person],y11List[person]), 
                            (x12List[person],y12List[person]), (x13List[person],y13List[person]), (x14List[person],y14List[person]), (x15List[person],y15List[person]))
                        max_queue[person].put(left_val)  # Use negative values for max priority queue
                        min_queue[person].put(-left_val)
                        max_queue_right[person].put(right_val)
                        min_queue_right[person].put(-right_val)
                        # If the queues exceed their maximum size, remove the highest/lowest element
                        if max_queue[person].qsize() >qsize :
                            max_queue[person].get()
                        if min_queue[person].qsize() > qsize:
                            min_queue[person].get() 
                            
                        if max_queue_right[person].qsize() >qsize :
                            max_queue_right[person].get()
                        if min_queue_right[person].qsize() > qsize:
                            min_queue_right[person].get()
                        #right_val = blinked((x10[person],y10[person]),(x11[person],y11[person]), 
                          #(x12[person],y12[person]), (x13[person],y13[person]), (x14[person],y14[person]), (x15[person],y15[person]))

                        max_average = sum(list(max_queue[person].queue)) / max_queue[person].qsize() if max_queue[person].qsize() > 0 else 0
                        min_average = -sum(list(min_queue[person].queue)) / min_queue[person].qsize() if min_queue[person].qsize() > 0 else 0
                        threshhold=min_average+0.6*(max_average-min_average)
                        left_blink[person]=detect(left_val, threshhold)
                        
                        max_average = sum(list(max_queue_right[person].queue)) / max_queue_right[person].qsize() if max_queue_right[person].qsize() > 0 else 0
                        min_average = -sum(list(min_queue_right[person].queue)) / min_queue_right[person].qsize() if min_queue_right[person].qsize() > 0 else 0
                        threshhold=min_average+0.6*(max_average-min_average)
                        right_blink[person]=detect(right_val, threshhold)
                        #print(left_blink[person], "   ", right_blink[person]);

                        if(left_blink[person] == 0 and right_blink[person] == 0):
                            sleep[person]+=1
                            #print(person," ",sleep[person])
                            if(sleep[person]>38):
                                status="SLEEPING !!!"
                                cv2.putText(frame, "Person" + str(person+1) +"Sleeping", (100*(person*1),150+(50*person)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                cv2.imshow('MediaPipe FaceMesh', frame)
                                color = (255,0,0)
                        else:
                            sleep[person] = 0
                        
                        
                        
                        ##########Exogenous Health Event##################
                        z1List[person] = mPointsList[person][93].z*wid  ### considered the depth (### Done)
                        z2List[person] = mPointsList[person][323].z*wid
                        z3List[person] = mPointsList[person][4].z*wid
                                                                
                        centroid = [(x1List[person]+x2List[person]+x3List[person])/3, (y1List[person]+y2List[person]+y3List[person])/3, (z1List[person]+z2List[person]+z3List[person])/3]
                        centroidx[person].append(centroid[0])
                        centroidy[person].append(centroid[1])
                        centroidz[person].append(centroid[2])
                        frames_observed=20

                                 
                        if len(centroidx[person]) > frames_observed:
                           centroidx[person].pop(0)
                        sdx = st.stdev(centroidx[person])
                               
                    
                        if len(centroidy[person]) > frames_observed:
                           centroidy[person].pop(0)
                        sdy = st.stdev(centroidy[person])
                        sdvxlist[person].append(sdx)
                        sdvylist[person].append(sdy)
                       
                       
                        if len(sdvxlist[person])>30: #30 here reprents CA threshhold
                           sdvxlist[person].pop(0)
                        if len(sdvylist[person])>30: #30 here reprents CA threshhold
                           sdvylist[person].pop(0)
                       
                        if len(centroidz[person]) > frames_observed:
                           centroidz[person].pop(0)
                        sdz = st.stdev(centroidz[person])
                        if len(centroidx[person])>2:
                            vx,vy,vz,v_magnitude   =velocities(centroidx[person],centroidy[person],centroidz[person],frames_observed)       
                            signsx=[]
                            for i,e in enumerate(vx):
                                if i==0:
                                    signsx.append(e)
                                elif signsx[-1]*e<0 and abs(e)>400:
                                    signsx.append(e)
    
                            signsy=[]
                            for i,e in enumerate(vy):

                                if i==0:
                                    signsy.append(e)
                                elif signsy[-1]*e<0 and abs(e)>400:
                                    signsy.append(e)
                            CA=max(len(signsx),len(signsy))
                            #print(CA)
                            if len(signsx)>=4 or len(signsy)>=4:
                                cv2.putText(frame, "Person" + str(person+1) +"EXOGENEOUS HEALTH EVENT", (100*(person*1),150+(50*person)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                cv2.imshow('MediaPipe FaceMesh', frame)
                                #print('EXOGENEOUS HEALTH EVENT 111111111111111111111111111')


                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Release resources
cap.release()
#out.release()
cv2.destroyAllWindows()
                  
