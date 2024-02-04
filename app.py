from flask import Flask, request, jsonify
import cv2
import dlib
import json
import numpy as np

app = Flask(__name__)

# Load the face detector model from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the facial landmarks predictor from dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variables for face movement detection
prev_face_center = None
face_movement_threshold = 20  # Adjust this threshold for better accuracy

# Variables for eye blink detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 4
ear_buffer = []

def detect_faces(image):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_detector(gray)

    return faces

def detect_blinks(eye_points, landmarks):
    left_eye = landmarks[eye_points[0]:eye_points[1]]
    right_eye = landmarks[eye_points[2]:eye_points[3]]

    # Calculate the eye aspect ratio
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Average the eye aspect ratio for both eyes
    ear = (left_ear + right_ear) / 2.0

    return ear

def eye_aspect_ratio(eye):
    # Calculate the distance between vertical eye landmarks manually
    A = np.sqrt((eye[1].x - eye[5].x) ** 2 + (eye[1].y - eye[5].y) ** 2)
    B = np.sqrt((eye[2].x - eye[4].x) ** 2 + (eye[2].y - eye[4].y) ** 2)

    # Calculate the distance between horizontal eye landmarks manually
    C = np.sqrt((eye[0].x - eye[3].x) ** 2 + (eye[0].y - eye[3].y) ** 2)

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def process_frame(frame):
    # Initialize json_response
    json_response = {"face_direction": "None", "eye_blink": False}

    global prev_face_center, ear_buffer

    # Detect faces
    faces = detect_faces(frame)

    if faces:  # Only proceed if faces are detected
        # Get the first detected face (assuming only one face in the frame)
        face = faces[0]

        # Get facial landmarks
        landmarks = predictor(frame, face)

        # Detect face movements
        face_center = (int((face.left() + face.right()) / 2), int((face.top() + face.bottom()) / 2))

        # Check if face moved significantly
        if prev_face_center is not None:
            # Check for specific face directions
            x_diff = face_center[0] - prev_face_center[0]
            y_diff = face_center[1] - prev_face_center[1]

            if abs(x_diff) > abs(y_diff):
                if x_diff > 0:
                    json_response["face_direction"] = "Right"
                else:
                    json_response["face_direction"] = "Left"
            else:
                if y_diff > 0:
                    json_response["face_direction"] = "Down"
                else:
                    json_response["face_direction"] = "Up"

        # Update previous face center
        prev_face_center = face_center

        # Detect eye blinks
        left_eye_points = [36, 42, 42, 48]
        right_eye_points = [42, 48, 36, 42]

        left_ear = detect_blinks(left_eye_points, landmarks.parts())
        right_ear = detect_blinks(right_eye_points, landmarks.parts())

        # Calculate the average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0

        # Add the current ear to the buffer
        ear_buffer.append(avg_ear)

        # Keep the buffer size within the defined limit
        ear_buffer = ear_buffer[-EYE_AR_CONSEC_FRAMES:]

        # Check for eye blink using the smoothed ear values
        if avg_ear < EYE_AR_THRESH and len(ear_buffer) >= EYE_AR_CONSEC_FRAMES:
            json_response["eye_blink"] = True

        # Log face direction and eye blink in the backend terminal
        print(f"Face Direction: {json_response['face_direction']}, Eye Blink: {json_response['eye_blink']}")

    return json_response

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "Empty file provided"}), 400

    # Read the image from the file
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Process the image and get face direction as JSON
    json_response = process_frame(image)

    return jsonify(json_response)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
