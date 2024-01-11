from flask import Flask, render_template, Response
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import dlib
import time
from flask_socketio import SocketIO
import numpy as np
import pygame

app = Flask(__name__)
socketio = SocketIO(app)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

pygame.mixer.init()
eyes_closed_sound = pygame.mixer.Sound('./music.wav')  # Replace with the path to your sound file

# Function to read frames from the camera
def get_frame():
    global camera, rawCapture, detector, predictor

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape_np = np.array([(part.x, part.y) for part in shape.parts()], dtype=np.int32)

            left_eye = shape_np[42:48]
            right_eye = shape_np[36:42]

            if len(left_eye) > 1 or len(right_eye) > 1:
                # Eyes are detected
                eyes_closed_sound.stop()
            else:
                # Eyes are not detected, play sound
                eyes_closed_sound.play()

            for (x, y) in left_eye:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            for (x, y) in right_eye:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

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
