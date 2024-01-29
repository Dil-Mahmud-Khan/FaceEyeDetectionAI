# import cv2
# import dlib

# def detect_face(image):
#     # Load the face detector model from dlib
#     detector = dlib.get_frontal_face_detector()

#     # Convert the image to grayscale for face detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale image
#     faces = detector(gray)

#     return faces

# def main():
#     # Open a video capture object
#     video_capture = cv2.VideoCapture(0)

#     # Variables to store previous face positions
#     prev_left, prev_right, prev_up, prev_down = False, False, False, False

#     while True:
#         # Read a frame from the video stream
#         ret, frame = video_capture.read()

#         # Detect faces in the frame
#         faces = detect_face(frame)

#         # Check if a face is detected
#         if len(faces) > 0:
#             face = faces[0]  # Assuming only one face is present

#             # Get face coordinates
#             x, y, w, h = face.left(), face.top(), face.width(), face.height()

#             # Determine face movements
#             moved_left = x < prev_left
#             moved_right = x + w > prev_right
#             moved_up = y < prev_up
#             moved_down = y + h > prev_down

#             # Update previous face positions
#             prev_left, prev_right, prev_up, prev_down = x, x + w, y, y + h

#             # Provide feedback based on face movements
#             if moved_left:
#                 print("Face moved right")
#             elif moved_right:
#                 print("Face moved left")
#             elif moved_up:
#                 print("Face moved up")
#             elif moved_down:
#                 print("Face moved down")

#             # Draw rectangles around detected faces for visual confirmation
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Display the frame with rectangles around faces
#         cv2.imshow('Video', frame)

#         # Break the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture object and close the window
#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
