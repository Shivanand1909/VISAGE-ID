# Import necessary libraries
import cv2                      # For webcam and image display
import numpy as np              # For numerical operations
import face_recognition         # For face detection and encoding
import os                       # For file and folder operations
from datetime import datetime   # For timestamping attendance

# Step 1: Load known images and names
images = []         # List to store image data
classNames = []     # List to store corresponding names

# Path to folder containing subfolders for each person
path = 'images'     # Folder structure: images/Name1/*.jpg, images/Name2/*.jpg, etc.

# Loop through each person's folder
for person in os.listdir(path):
    person_folder = os.path.join(path, person)
    if os.path.isdir(person_folder):
        for file in os.listdir(person_folder):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(person_folder, file)
                img = cv2.imread(img_path)
                images.append(img)
                classNames.append(person)  # Folder name is used as person's name

# Step 2: Encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])         # Store first encoding found
    return encodeList

encodeListKnown = findEncodings(images)
print('✅ Face encoding complete.')

# Step 3: Mark attendance with timestamp
def markAttendance(name):
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')       # Format: 2025-09-11
    timeString = now.strftime('%H:%M:%S')       # Format: 21:57:00
    entry = f'{name},{timeString},{dateString}\n'

    # Create file if it doesn't exist
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w') as f:
            f.write('Name,Time,Date\n')

    # Append new entry every time code runs
    with open('attendance.csv', 'a') as f:
        f.write(entry)
    print(f'📝 Attendance marked for {name} at {timeString} on {dateString}')

# Step 4: Start webcam and recognize faces
cap = cv2.VideoCapture(0)  # Open webcam
attendance_given = set()   # Track names already marked in this run

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame for faster processing
    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgbSmallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    facesCurrentFrame = face_recognition.face_locations(rgbSmallFrame)
    encodingsCurrentFrame = face_recognition.face_encodings(rgbSmallFrame, facesCurrentFrame)

    # Compare each detected face with known encodings
    for encodeFace, faceLoc in zip(encodingsCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDist)

        # If match is found and confidence is high
        if matches[matchIndex] and faceDist[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()  # Convert name to uppercase

            # Only mark attendance once per run
            if name not in attendance_given:
                markAttendance(name)
                attendance_given.add(name)

            # Draw rectangle and name on screen
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Scale back to original frame size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow('Face Attendance System', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()