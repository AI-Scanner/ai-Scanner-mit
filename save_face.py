import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start video capture
cap = cv2.VideoCapture(0)
time.sleep(2)  # Give the camera time to initialize

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

saved_faces_count = 0  # Store the number of saved faces
max_samples = 5  # Maximum number of samples to save per person
face_saved = False  # Set a flag to prevent multiple saves for the same face detection

# Create a directory to save face data if it doesn't exist
data_dir = 'face_data'
os.makedirs(data_dir, exist_ok=True)

# Name of the person to save
person_name = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks_list = []

            # Extract the 468 landmarks and save them as a list of [x, y, z] points
            for lm in face_landmarks.landmark:
                landmarks_list.append([lm.x, lm.y, lm.z])

            # Draw the face landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,  # Tesselation drawing connections
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,  # Contour drawing connections
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Only save the face when 's' is pressed and the name is set
            if not face_saved and person_name:
                if saved_faces_count < max_samples:
                    np.save(os.path.join(data_dir, f"{person_name}_{saved_faces_count}.npy"), np.array(landmarks_list))
                    print(f"{person_name}'s face sample {saved_faces_count + 1} saved!")
                    saved_faces_count += 1  # Increment the saved face count
                    face_saved = True  # Prevent repeated saves for the same face detection

    # Prompt for the person's name when 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c') and not person_name:
        person_name = input("Enter the name of the person: ")
        print(f"Ready to save samples for {person_name}. Press 's' to save samples.")

    # Reset the flag when 'n' is pressed to allow saving the next face
    if cv2.waitKey(1) & 0xFF == ord('n'):
        face_saved = False
        print(f"Ready to save the next sample for {person_name}.")

    # Reset saved faces count when a new person is being saved
    if saved_faces_count >= max_samples:
        print(f"Saved {max_samples} samples for {person_name}. Press 'c' to enter a new name.")
        person_name = ""
        saved_faces_count = 0

    # Display the frame
    cv2.imshow('Save Face', frame)

    # Press 'q' to quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows if 'q' is pressed
cap.release()
cv2.destroyAllWindows()
