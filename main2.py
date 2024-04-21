import numpy as np
#"C:\gun"
import cv2

import imutils

import datetime
#image_dir='../content/image-1 gun.jfif'
#annot_dir='../input/weapon-detection-dataset/Sohas_weapon-Detection/annotations/xmls/'



gun_cascade = cv2.CascadeClassifier( "C:cascade1.xml")

camera = cv2.VideoCapture ("C:\gun")
#image_dir='../content/image-1 gun.jfif'
firstFrame = None

gun_exist = False

while True:

    ret, frame = camera.read()

    if frame is None:

        break

    frame = imutils.resize(frame, width=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))

    if len(gun) > 0:

        gun_exist = True

    for (x, y, w, h) in gun:

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]

        roi_color = frame[y:y + h, x:x + w]

    if firstFrame is None:

        firstFrame = gray

        continue

    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),

                (10, frame.shape[0] - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.35, (0, 0, 255), 1)

    if gun_exist:

        print("Guns detected")

        plt.imshow(frame)

        break

    else:

        cv2.imshow("Security Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
         print("camera")
         break

camera.release()
cv2.destroyAllWindows()
#import cv2
#import numpy as np

# Initialize video capture
cap=cv2.VideoCapture(0)  # You can replace 0 with the video file path for offline tracking

# Read the first frame
ret, frame1 = cap.read()

# Check if the first frame is read successfully
if not ret:
    print("Error: Couldn't read the first frame from the camera.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    # Check if the next frame is read successfully
    if not ret:
        print("Error: Couldn't read the next frame from the camera.")
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude and direction
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set the hue value based on the direction of the flow
    hsv[..., 0] = ang * 180 / np.pi / 2

    # Set the value based on the magnitude of the flow
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display the resulting frame
    cv2.imshow('Motion Tracking', bgr)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update the previous frame
    prvs = next_frame

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()