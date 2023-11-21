						x1List = []
						y1List = []
						
						
						x23List = []
						y23List = []
						
						x1List= [0]*numberOfPerson
						y1List= [0]*numberOfPerson
						x2List= [0]*numberOfPerson
						y2List= [0]*numberOfPerson
						x3List= [0]*numberOfPerson
						y3List= [0]*numberOfPerson
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

						
						
						
						x1List[person] = mPointsList[person][93].x*wid
						x2List[person] = mPointsList[person][323].x*wid
						x3List[person] = mPointsList[person][4].x*wid
						y1List[person] = mPointsList[person][93].y*heigh
						y2List[person] = mPointsList[person][323].y*heigh
						y3List[person] = mPointsList[person][4].y*heigh

                   
                        
                    
						x4List[person] = mPointsList[person][33].x*wid
						y4List[person] = mPointsList[person][33].y*heigh)
						x5List[person] = mPointsList[person][160].x*wid)
						y5List[person] = mPointsList[person][160].y*heigh)
						x6List[person] = mPointsList[person][158].x*wid)
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
                            yawn_count[person]+=1
                            #print(yawn_count)
                            if(yawn_count[person]>15):
                                    print('Person'+ person +'yawning')
                                    #self.label1_3.setText("Yawning");
                                    cv2.putText(image, "Yawning", (100*(person+1),300*(person+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),3)
                        
                        else:
                            yawn_count[person]=0   
