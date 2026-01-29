import cv2
import time
import config
import numpy as np
from utils.picamera_utils import is_raspberry_camera, get_picamera
from utils.drawing_utils import draw_face_rectangles, draw_emotion_text, draw_emotion_image


IS_RASPI_CAMERA = is_raspberry_camera()
print(f"Using raspi camera: {IS_RASPI_CAMERA}")

if __name__ == "__main__":
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(config.FACE_MODEL_PATH)
    emotion_net = cv2.dnn.readNetFromONNX(config.EMOTION_MODEL_PATH)  # Placeholder for emotion recognition model loading if needed
    fps = 0 # initial FPS value
    
    try:
        # Init camera
        if IS_RASPI_CAMERA:
            cap = get_picamera(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
            cap.start()
        else:
            cap = cv2.VideoCapture(config.CAMERA_DEVICE_ID)
            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.IMAGE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.IMAGE_HEIGHT)

        emotion_images = {}
        for emotion in config.EMOTIONS:
            img_path = f"{config.EMOTION_ICONS_PATH}/{emotion}.jpg"
            img = cv2.imread(img_path)
            if img is not None:
                emotion_images[emotion] = cv2.resize(img, (config.EMOTION_IMAGE_SIZE, config.EMOTION_IMAGE_SIZE))
    
        #Main loop
        while True:
            start_time = time.time()

            # Capture frame
            if IS_RASPI_CAMERA:
                frame = cap.capture_array()
            else:
                _, frame = cap.read()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)

                emotion_net.setInput(blob)
                preds = emotion_net.forward()

                emotion_idx = np.argmax(preds[0])
                emotion_text = config.EMOTIONS[emotion_idx]

                frame = draw_face_rectangles(frame, faces)
                frame = draw_emotion_text(frame, emotion_text, (x, y-10))
                '''
                print(f"Эмоция: {emotion_text}")'''
            
            if len(faces) > 0:
                emotion_image = emotion_images.get(emotion_text, None)
                frame = draw_emotion_image(frame, emotion_image, (10, 10))

            # Display the resulting frame
            cv2.imshow("Emotion Project", frame)

            # Calculate FPS
            end_time = time.time()
            dt = end_time - start_time
            if dt > 0:
                fps = 1.0 / dt

            # Exit on 'ESC' key
            if cv2.waitKey(1) == 27:
                break
                
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Release resources
        cv2.destroyAllWindows()
        if 'cap' in locals():
            cap.close() if IS_RASPI_CAMERA else cap.release()