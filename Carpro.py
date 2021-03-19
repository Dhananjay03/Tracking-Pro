import cv2
#our img
#img_file = "car4.jpg"
#video = cv2.VideoCapture('Tesla Autopilot Dashcam Compilation 2018 Version.mp4')
video = cv2.VideoCapture('Zombie Pedestrians walking into the road.mp4')

#our pre-trained classifiers
car_tracker_file = "car_detector.xml"
Pedestrian_tracker_file = "Pedestrians_detector.xml"

#create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)

Pedestrian_tracker = cv2.CascadeClassifier(Pedestrian_tracker_file)

#Run forever util car stops
while True:
    #read the current frame from video
    (read_successful, frame )= video.read()

    #safe coding.
    if read_successful:
        #must convert to grayscale and pedestrains
        grayscaled_frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
         break 

 
    #detects cars and pedestrains
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = Pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #to draw a rectangle around the cars.
    for (x, y, w, h) in cars: 
        cv2.rectangle(frame, (x+1,y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    # to draw a rectangle around pedestrains  
    for(x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+w), (0,255,255), 2)   

    #print(cars)
    #display the image with faces spotted
    cv2.imshow('self Driving Cars',frame)

    
    #dont autoclose (wait here in the code and listen for a key press)
    key =cv2.waitKey(10)
    
    #stop if Q is pressed.
    if key==81 or key==113:
        break
    #release the videoCapture object    
video.release()    




"""
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)




#print where the car is 


print(cars)


# print(car1)







"""


print("code completed")  