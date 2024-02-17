import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the Video
cap = cv2.VideoCapture('C:/Users/Debarun/Downloads/WHATSAAP ASSIGNMENT.mp4')

# Create the color Finder object
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 19, 'smin': 95, 'vmin': 0, 'hmax': 46, 'smax': 255, 'vmax': 252}


prediction = False
bounce = 0
lhb = 0
rhb = 0
cross = 0
left_dist = 0.0
right_dist = 0.0
prev_y = 0
prev_y_2 = 0

lw_x = []
rw_x = []
lw_y = []
rw_y = []
frame_counter = 0


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # Grab the image

        success, img = cap.read()
        #img = cv2.imread("C:/Users/Debarun/Downloads/Screenshot 2024-02-17 011641.png")
        img = img[0:700, :] 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        results = pose.process(img)
    
        # Recolor back to BGR
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
        except:
            pass        
          
        # Find the Color Ball
        imgColor, mask = myColorFinder.update(img, hsvVals)

        # Find location of the Ball
        imgContours, contours = cvzone.findContours(img, mask, minArea=400)

        mp_drawing.draw_landmarks(imgContours, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )             

        lw_x.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * 1000)
        lw_y.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * 1000)
        rw_x.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * 1000)
        rw_y.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * 1000)

        frame_counter = frame_counter + 1
        
        if contours:
            
            cx, cy = contours[0]['center']
            cent = (cx, cy)
            cv2.circle(imgContours, cent, 5, (0, 255, 0), cv2.FILLED )

            
                    
            
            for i in contours:
                if (cy < prev_y) and (prev_y > prev_y_2) :
                    bounce = bounce + 1
                    j = frame_counter - 15
                    left_dist = math.sqrt(((cx*2)-lw_x[j])**2 + (cy-lw_y[j])**2 )
                    right_dist = math.sqrt(((cx*2)-rw_x[j])**2 + (cy-rw_y[j])**2 )

                    if (left_dist < right_dist):
                        lhb = lhb + 1
                        

                    elif (right_dist < left_dist):
                        rhb = rhb + 1                       

                    

                    
                prev_y_2 = prev_y
                prev_y = cy
                
                
            
        
        

            
            

        # Display
        imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)

        cv2.putText(imgContours, f"Dribbles: {(int(bounce))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(imgContours, f"Right Hand Dribbles: {rhb}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(imgContours, f"Left Hand Dribbles: {lhb}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        #cv2.imshow("Image", img)

        cv2.imshow("ImageColor", imgContours)
        cv2.waitKey(30)

