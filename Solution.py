import cv2
import numpy as np
import imageio

def get_contours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:
            # cv2.drawContours(imgContour, cnt, -20, (0, 0, 0), 3)

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 0), 5)
            # cv2.putText(imgContour, 'Area:' + str(int(area)), (x +w +20, y + 20),
            # cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
            # cv2.putText(imgContour, 'Approx:' + str(int(len(approx))), (x + w + 20, y + 20),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
            if len(approx) == 3:
                # cv2.drawContours(imgContour, cnt, -1, (7, 183, 115), 7)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (7, 183, 115), 5)
                cv2.putText(imgContour, 'Triangle', (x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (7, 183, 115), 2)
            elif len(approx) == 4:
                # cv2.drawContours(imgContour, cnt, -1, (28, 112, 197), 7)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (28, 112, 197), 5)
                cv2.putText(imgContour, 'Rectangle', (x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (28, 112, 197), 2)
            else:
                # cv2.drawContours(imgContour, cnt, -1, (201, 10, 60), 7)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (201, 10, 60), 5)
                cv2.putText(imgContour, 'Circle', (x + w + 20, y + 45),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (201, 10, 60), 2)


reader = imageio.get_reader('Resources/Test task1_video.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('Task_1.mp4', fps = fps)
for i, frame in enumerate(reader):

    # imageio reads frames in RGB, so we should convert it to BGR for OpenCV
    imgBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    imgBlur = cv2.GaussianBlur(imgBGR, (15, 15), 20)

    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([55, 25, 30])
    upper_hsv = np.array([92, 103, 253])
    mask = cv2.inRange(imgHSV, lower_hsv, upper_hsv)
    
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(mask, kernel, iterations=1)

    imgCanny = cv2.Canny(imgDil, 60, 60)

    get_contours(imgCanny, frame)
    
    writer.append_data(frame)
writer.close()