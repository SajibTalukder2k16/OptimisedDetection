import cv2
import mediapipe as mp

def detect_poses(image, rois):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    all_landmarks = []

    # Process each ROI separately
    for roi in rois:
        roi_image = image_rgb[roi[1]:roi[3], roi[0]:roi[2]]

        # Make a detection
        results = pose.process(roi_image)

        # Draw landmarks on the image
        if results.pose_landmarks:
            landmarks = [(int(landmark.x * roi_image.shape[1]), int(landmark.y * roi_image.shape[0])) for landmark in results.pose_landmarks.landmark]
            all_landmarks.append(landmarks)

    return all_landmarks

# Example using webcam
cap = cv2.VideoCapture(0)

# Define ROIs (x1, y1, x2, y2)
rois = [(0, 0, 320, 480), (320, 0, 640, 480)]  # Example: Two ROIs, left and right

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect poses in each ROI
    landmarks_list = detect_poses(frame, rois)

    # Draw landmarks on the original frame
    for landmarks in landmarks_list:
        for landmark in landmarks:
            cv2.circle(frame, landmark, 5, (0, 255, 0), -1)

    # Display the result
    cv2.imshow('Multi-Person Pose Detection', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
