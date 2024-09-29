import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Directory to save face data
data_dir = 'face_data'

# Create the directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Function to load all saved face data
def load_saved_data():
    saved_data = {}
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            person_name = file.replace("_landmarks.npy", "")
            saved_data[person_name] = np.load(os.path.join(data_dir, file))
    return saved_data

# Function to calculate the distance between two sets of face landmarks
def calculate_distance(landmarks1, landmarks2):
    # Reshape landmarks to (468, 3)
    landmarks1 = np.array(landmarks1).reshape(468, 3)
    landmarks2 = np.array(landmarks2).reshape(468, 3)
    
    # Normalize landmarks by the distance between the eyes
    eye_distance = np.linalg.norm(landmarks1[37] - landmarks1[40])  # Example: using left and right eye landmarks
    landmarks1 /= eye_distance
    landmarks2 /= eye_distance

    return np.linalg.norm(landmarks1 - landmarks2)


# Function to identify a person based on the landmarks
def identify_person(landmarks_list, saved_data):
    min_dist = float('inf')
    identified_person = "Unknown"
    
    for person_name, saved_landmark_list in saved_data.items():
        dist = calculate_distance(landmarks_list, saved_landmark_list)
        if dist < min_dist:
            min_dist = dist
            identified_person = person_name
    
    return identified_person

# Start video capture
cap = cv2.VideoCapture(0)

# Load saved data initially
saved_data = load_saved_data()

last_identified_person = "Unknown"  # Keep track of the last identified person

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect face landmarks
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks_list = []

            # Extract the 468 landmarks
            for lm in face_landmarks.landmark:
                landmarks_list.append([lm.x, lm.y, lm.z])

            # Identify the person based on the landmarks
            identified_person = identify_person(landmarks_list, saved_data)

            # Draw the face mesh contours on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )

            # Draw a rectangle around the face
            h, w, _ = frame.shape
            x_min = int(min(lm.x for lm in face_landmarks.landmark) * w)
            x_max = int(max(lm.x for lm in face_landmarks.landmark) * w)
            y_min = int(min(lm.y for lm in face_landmarks.landmark) * h)
            y_max = int(max(lm.y for lm in face_landmarks.landmark) * h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Label the face
            label = identified_person
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print when the identification changes
            if identified_person != last_identified_person:
                print(f"Identified person: {identified_person}")
                last_identified_person = identified_person

    else:
        # Draw "Unknown" if no face is detected
        cv2.putText(frame, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
