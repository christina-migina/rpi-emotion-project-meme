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


def draw_emotion_image(image, emotion_image, position):
    if emotion_image is None:
        return image
    
    x, y = position
    h, w = emotion_image.shape[:2]
    
    if y < 0 or x < 0 or y + h > image.shape[0] or x + w > image.shape[1]:
        return image

    if image.shape[2] == 4 and emotion_image.shape[2] == 3:
        emotion_image = cv2.cvtColor(emotion_image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] == 3 and emotion_image.shape[2] == 4:
        emotion_image = cv2.cvtColor(emotion_image, cv2.COLOR_BGRA2BGR)

    image[y:y+h, x:x+w] = emotion_image
    return image