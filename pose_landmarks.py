import cv2
import mediapipe as mp

def detect_pose_landmarks(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make a detection
    results = pose.process(image_rgb)

    # Draw landmarks on the image
    if results.pose_landmarks:
        print(results.pose_landmarks)
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return image

# Example using webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect pose landmarks
    frame_with_landmarks = detect_pose_landmarks(frame)
    frame_with_landmarks = cv2.resize(frame,(1080,720))

    # Display the result
    cv2.imshow('Pose Landmarks Detection', frame_with_landmarks)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
