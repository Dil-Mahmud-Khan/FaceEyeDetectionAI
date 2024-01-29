# import cv2
# import dlib

# def detect_faces(image, detector):
#     # Convert the image to grayscale for face detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale image
#     faces = detector(gray)

#     return faces

# def detect_blinks(eye_points, landmarks):
#     left_eye = landmarks[eye_points[0]:eye_points[1]]
#     right_eye = landmarks[eye_points[2]:eye_points[3]]

#     # Calculate the eye aspect ratio
#     left_ear = eye_aspect_ratio(left_eye)
#     right_ear = eye_aspect_ratio(right_eye)

#     # Average the eye aspect ratio for both eyes
#     ear = (left_ear + right_ear) / 2.0

#     return ear

# def eye_aspect_ratio(eye):
#     # Calculate the distance between vertical eye landmarks
#     A = dist(eye[1], eye[5])
#     B = dist(eye[2], eye[4])

#     # Calculate the distance between horizontal eye landmarks
#     C = dist(eye[0], eye[3])

#     # Calculate the eye aspect ratio
#     ear = (A + B) / (2.0 * C)

#     return ear

# def dist(p1, p2):
#     # Calculate the Euclidean distance between two points
#     return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

# def main():
#     # Open a video capture object
#     video_capture = cv2.VideoCapture(0)

#     # Load the face detector model from dlib
#     face_detector = dlib.get_frontal_face_detector()

#     # Load the facial landmarks predictor from dlib
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#     # Variables for face movement detection
#     prev_left, prev_right, prev_up, prev_down = False, False, False, False

#     # Variables for eye blink detection
#     EYE_AR_THRESH = 0.3
#     EYE_AR_CONSEC_FRAMES = 3
#     COUNTER = 0

#     try:
#         while True:
#             # Read a frame from the video stream
#             ret, frame = video_capture.read()

#             # Detect faces
#             faces = detect_faces(frame, face_detector)

#             for face in faces:
#                 # Get facial landmarks
#                 landmarks = predictor(frame, face)

#                 # Detect face movements
#                 x, y, w, h = face.left(), face.top(), face.width(), face.height()
#                 moved_left = x < prev_left
#                 moved_right = x + w > prev_right
#                 moved_up = y < prev_up
#                 moved_down = y + h > prev_down

#                 # Update previous face positions
#                 prev_left, prev_right, prev_up, prev_down = x, x + w, y, y + h

#                 # Provide feedback based on face movements
#                 if moved_left:
#                     print("Face moved right")
#                 elif moved_right:
#                     print("Face moved left")
#                 elif moved_up:
#                     print("Face moved up")
#                 elif moved_down:
#                     print("Face moved down")

#                 # Detect eye blinks
#                 left_eye_points = [36, 42, 42, 48]
#                 right_eye_points = [42, 48, 36, 42]

#                 left_ear = detect_blinks(left_eye_points, landmarks.parts())
#                 right_ear = detect_blinks(right_eye_points, landmarks.parts())

#                 if left_ear < EYE_AR_THRESH and right_ear < EYE_AR_THRESH:
#                     COUNTER += 1
#                 else:
#                     if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                         print("Eye blinked")
#                     COUNTER = 0

#                 # Draw rectangles around detected faces for visual confirmation
#                 cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

#             # Display the frame with rectangles around faces
#             cv2.imshow('Video', frame)

#             # Break the loop when 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     except Exception as e:
#         print(f"An error occurred: {e}")

#     finally:
#         # Release the video capture object and close the window
#         video_capture.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
