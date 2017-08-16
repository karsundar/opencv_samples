import numpy as np
import cv2, os
from matplotlib import pyplot as plt

def main():
    cap = cv2.VideoCapture(0)
    path = '/home/karthik/samples/opencv_samples'

    i = 2
    print('/home/karthik/samples/opencv_samples/{}'.format(i))
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            print('video capture failed to read.')
            break;

        face_cascade = cv2.CascadeClassifier(os.path.join(path, 'data/haarcascades/haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(os.path.join(path, 'data/haarcascades/haarcascade_eye.xml'))

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        plt.axis("off")
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.imsave('/home/karthik/samples/opencv_samples/data/training/{}.jpeg'.format(i), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), format='JPEG')
        i = i+1

        #plt.show(block=False)
        plt.pause(0.00001)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

if __name__ == '__main__':
    main()