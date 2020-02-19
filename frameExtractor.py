import cv2
import os

def frameExtractor(folder_name, video_path, medium):
    cap = cv2.VideoCapture(video_path)
    i = 0
    dirname = folder_name
    #medium = 0
    os.mkdir(dirname)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if(medium):
            frame =  cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        file = os.path.join(dirname, str(i) + '.jpg')
        cv2.imwrite(file, frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()