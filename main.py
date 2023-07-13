import cv2
import numpy as np

# Constants for camera calibration
KNOWN_DISTANCE = 100  # Distance in centimeters from the camera to the object (e.g., person)
KNOWN_WIDTH = 45  # Width of the object (e.g., person's average shoulder width) in centimeters

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Calculate the distance to the object based on the known width and pixel width of the detected face
def estimate_distance(known_width, face_width):
    return abs((known_width * cap.get(3)) / (2 * face_width * np.tan(cap.get(3) * 0.5 * np.pi / 180)))

# Set the alarm threshold
ALARM_THRESHOLD = 50  # Distance threshold in centimeters

# Loop through the frames
while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the distance to the person
        distance = estimate_distance(KNOWN_WIDTH, w)

        # Display the distance on the frame
        distance_text = f"Distance: {distance:.2f} cm"
        cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if distance is below the threshold
        if distance < ALARM_THRESHOLD:
            print("Person is too close!")  # Print a message indicating that the person is too close

    # Display the resulting frame
    cv2.imshow('Person Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
