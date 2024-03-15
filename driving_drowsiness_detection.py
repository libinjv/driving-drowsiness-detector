# Import neccessary libraries
import cv2
import dlib
from pygame import mixer
import imutils
from imutils import face_utils
from scipy.spatial import distance
import time

# Initialize pygame mixer and load the alarm
mixer.init()
mixer.music.load("alarm.wav")


# define a function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
	vert_eyelid_dist= distance.euclidean(eye[1], eye[5])
	vert_eye_dist = distance.euclidean(eye[2], eye[4])
	eye_width = distance.euclidean(eye[0], eye[3])
	ear = ( vert_eyelid_dist + vert_eye_dist) / (2.0 * eye_width)
	return ear

# setting threshold for eye aspect ratio	
ear_thresh = 0.26

# Initializing face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Extract the starting and ending indices of landmarks for left and right eye 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Capturing Video
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

#This variable is to counts consecutive frames with drowsiness
flag=0

# Get the frames per second (fps) of the video capture
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate frame check based on fps
if fps == 30:
    frame_check = 20
elif fps == 60:
    frame_check = 10
else:
    # Default to 30 fps if fps is not recognized
    frame_check = 20


text_on = False  
start_time=time.time()

# Main Loop:
while True:
	
	ret, frame=cap.read()
	# resizing the frame to 450 pixels for faster processing
	frame = imutils.resize(frame, width=450)
	# converting to grayscale image to simplify image processing and reduce computational complexity
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detected faces in images are stored to subject variable
	subjects = face_detector(gray, 0)
	
 
	# This loop iterates over the detected faces stored in the subjects variable
	for subject in subjects:
		# this predict the locations of facial landmarks for a detected face in the input grayscale image.
		shape = landmark_predictor(gray, subject)
		# converting shape to Numpy array 
		shape = face_utils.shape_to_np(shape)

		# Extractingh the facial landmarks corresponding to the left and right eyes from the shape array.
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
  
		# calculating the average of EAR from both eyes.
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull of the landmarks for both the left and right eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
  
		# draw the convex hulls of the left and right eyes onto the frame image.
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
  

		if ear < ear_thresh:
			flag += 1
			print (flag)
   
			if flag >= frame_check:
				#if drowsiness is detected it display this warning message and plays the loaded alarm
				cv2.putText(frame, "      WARNING!!--Drowsiness Detected      ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
				# to make take a break message blink:
				current_time=time.time()
				elapsed_time=current_time-start_time
				
				if elapsed_time %1 <0.5:
					text_on= True
				else:
					text_on=False
				if text_on:
					cv2.putText(frame, "               TAKE A BREAK              ", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 255), 2)
    	
					
				mixer.music.play()
				overlay=frame.copy()

				# creating a red overlay on the enitre video fram to highlight the warning message				
				cv2.rectangle(overlay,(0,0),(frame.shape[1],frame.shape[0]),(0,0,255),-1)
				alpha=0.2
				cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
    
		else:
			flag = 0

	# Displays the processed video frame 
	cv2.imshow("Driving Drowsiness Detector", frame)
 
	# assign 'q' on keyboard to quit or stop the program
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# releasing system resources associated with video capture and closing all OpenCV windows.
cv2.destroyAllWindows()
cap.release() 