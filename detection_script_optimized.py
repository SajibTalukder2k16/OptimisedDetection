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
status = "Status: "

import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces=5)
#################################

numberOfPerson = 2;

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
yawn_count = [0]*numberOfPerson
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






# Define codec and VideoWriter object to save the output video

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
    
cap = cv2.VideoCapture(0)



while cap.isOpened():
    
            ret, frame = cap.read()
            
            if not ret:
                break
            #frame=cv2.flip(frame,-1)
            wid=720
            heigh=480
            frame=cv2.resize(frame, (wid,heigh))
           
            if ret:
                # Convert the OpenCV image to a QImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #cv2.imshow('MediaPipe FaceMesh', frame)

                # Process the frame with FaceMesh
                results = face_mesh.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                

                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        # Draw landmarks on the frame
                        mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS, 
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255, 150), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255,150), thickness=1))
                        cv2.imshow('MediaPipe FaceMesh', frame)
                    
                if(results.multi_face_landmarks != None):
                    mPoints = []
                    detected_face = len(results.multi_face_landmarks);
                    for person in range(detected_face):
                        print("Person", person+1)
                        
                

                        if(len(results.multi_face_landmarks)>0):
                            if len(results.multi_face_landmarks)!=number_of_faces:
                                #print('1'*50)
                                max_queue.queue.clear()
                                min_queue.queue.clear()
                                number_of_faces=1
                            #mpoints1 =results.multi_face_landmarks[0].landmark
                            #mpoints2 = mpoints1
                            
                            mPoints.append(results.multi_face_landmarks[person].landmark)
                            print("Mmmm: ",len(mPoints))
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
                                #print("HA1= " +str(HA1))
                           
                                if HA1 <150 :
                                    heart_attack1+=1
                                    
                                    i=0
                                    if(heart_attack1>8):
                                        #print("heart_attack1 = " + str( heart_attack1))
                                        i+=1
                                        #self.label1_3.setText("Heart Attack");
                                        #cv2.putText(image, "Heart Attack", (500,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                                        #cv2.imshow('MediaPipe FaceMesh', image)
                                        
                                        #if i==1:
                                            #message = client.messages.create(to='+12028267898',from_="+19705358298",body=" Megan was distracted while driving")             
                                else:
                                  heart_attack1=0
                                  #self.label1_3.setText("")
    
    
                            
                            
                        
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
                        
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Release resources
cap.release()
#out.release()
cv2.destroyAllWindows()
                  
