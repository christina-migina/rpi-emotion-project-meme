import cv2
import config

def visualize_fps(image, fps: int):
    fps_text = 'FPS = {:.1f}'.format(fps)

    cv2.putText(
        image, 
        fps_text, 
        config.TEXT_MARGIN,
        cv2.FONT_HERSHEY_PLAIN,
        config.FONT_SIZE,
        config.TEXT_COLOR,
        config.FONT_THICKNESS
    )

    return image

def draw_face_rectangles(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            config.FACE_RECT_COLOR,
            config.FACE_RECT_THICKNESS
        )
    return image

def draw_emotion_text(image, text, position):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_PLAIN,
        config.FONT_SIZE,
        config.TEXT_COLOR,
        config.FONT_THICKNESS
    )
    return image