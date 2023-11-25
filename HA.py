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