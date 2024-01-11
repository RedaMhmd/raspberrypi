from flask import Flask, render_template, Response
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import dlib
import time
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

time.sleep(0.1)

total_eyes_closed_time = 0  # Total time when eyes are closed
eyes_closed_start_time = None
eyes_closed_threshold = 1  # Set the threshold for considering eyes closed (you may adjust this)

# Function to read frames from the camera
def get_frame():
    global camera, rawCapture, detector, predictor, eyes_closed_start_time, total_eyes_closed_time

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use dlib's face detector
        faces = detector(gray)
        
        # Reset the eyes_closed_start_time if no face is detected
        if not faces:
            eyes_closed_start_time = None

        for face in faces:
            # Use dlib's shape predictor to get facial landmarks
            shape = predictor(gray, face)
            shape = dlib.shape_to_np(shape)

            # Extract the coordinates of the eyes (adjust indices based on your specific facial landmarks)
            left_eye = shape[42:48]
            right_eye = shape[36:42]

            # Draw rectangles around the eyes for visualization
            for (x, y) in left_eye:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Check if eyes are detected
            if left_eye and right_eye:
                # Eyes are detected, measure the duration of open eyes
                if eyes_closed_start_time is not None:
                    eyes_open_duration = time.time() - eyes_closed_start_time
                    socketio.emit('update_eyes_open_duration', {'duration': eyes_open_duration})
                    eyes_closed_start_time = None

            else:
                # Eyes are not detected, measure the duration of closed eyes
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
                else:
                    eyes_closed_duration = time.time() - eyes_closed_start_time
                    total_eyes_closed_time += eyes_closed_duration
                    socketio.emit('update_eyes_closed_duration', {'duration': eyes_closed_duration})
                    socketio.emit('update_total_eyes_closed_time', {'duration': total_eyes_closed_time})
                    eyes_closed_start_time = time.time()

        _, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        rawCapture.truncate(0)

# Route for streaming video with face and eye detection
@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

