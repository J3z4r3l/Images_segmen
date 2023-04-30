import cv2
import numpy as np
import time

IMG_ROW_RES = 480
IMG_COL_RES = 640
PUBLISH_TIME = 10

def init_camera():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, IMG_COL_RES)
    video_capture.set(4, IMG_ROW_RES)
    return video_capture

def acquire_image(video_capture):
    ret, bgr_video = video_capture.read()
    scaled_rgb_frame = cv2.resize(bgr_video, (0, 0), fx=0.25, fy=0.25)[:, :, ::-1]
    return bgr_video, scaled_rgb_frame

def show_frame(name, frame):
    cv2.imshow(name, frame)

last_publication_time = 0.0
video_capture = init_camera()

while True:
    bgr_video, _ = acquire_image(video_capture)

    if time.time() - last_publication_time >= PUBLISH_TIME:
        try:
            print("No remote action needed ...")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
        last_publication_time = time.time()
     
    to_zero = cv2.threshold(bgr_video, 100, 255, cv2.THRESH_TOZERO)[1]
    gray_video = cv2.cvtColor(bgr_video, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray_video, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    _, th_bin_img = cv2.threshold(gray_video, 124, 255, cv2.THRESH_BINARY_INV)
    erosion_kernel = np.ones((15, 15), np.uint8)
    ero_img = cv2.erode(th_bin_img, erosion_kernel, iterations=1)
    ret, markers = cv2.connectedComponents(ero_img)
    watershed = cv2.watershed(bgr_video, markers)

    show_frame('RGB image', bgr_video)
    show_frame('Gray boards', adaptive)
    show_frame('Gray level image', cv2.applyColorMap(to_zero.astype(np.uint8), cv2.COLORMAP_JET))
    show_frame('Gsd', cv2.applyColorMap(watershed.astype(np.uint8), cv2.COLORMAP_CIVIDIS))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
