from flask import Flask, render_template, Response
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

time.sleep(0.1)

total_eyes_closed_time = 0  # Total time when eyes are closed
eyes_closed_start_time = None
eyes_closed_threshold = 1  # Set the threshold for considering eyes closed (you may adjust this)

# Function to read frames from the camera
def get_frame():
    global camera, rawCapture, faceCascade, eyesCascade, eyes_closed_start_time, total_eyes_closed_time

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Reset the eyes_closed_start_time if no face is detected
        if not len(faces):
            eyes_closed_start_time = None

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            eyes = eyesCascade.detectMultiScale(roi_gray)
            
            # Check if eyes are detected
            if len(eyes) > 0:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (100, 255, 255), 2)
                
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
                    total_eyes_closed_time += time.time() - eyes_closed_start_time
                    eyes_closed_start_time = time.time()

        # Emit total_eyes_closed_time to the webpage
        socketio.emit('update_total_eyes_closed_time', {'duration': total_eyes_closed_time})

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

