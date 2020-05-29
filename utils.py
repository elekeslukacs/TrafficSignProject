import cv2
import numpy as np
import tkinter
from tkinter import filedialog
import os


# Color filters
def detectRed(imgHsv):
    lower_red = np.array([0, 130, 75])
    upper_red = np.array([20, 255, 255])
    mask1 = cv2.inRange(imgHsv, lower_red, upper_red)

    lower_red = np.array([150, 125, 20])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(imgHsv, lower_red, upper_red)

    mask_red = mask1 + mask2
    return mask_red


def detectBlue(imgHsv):
    lower_blue = np.array([94, 127, 20])
    upper_blue = np.array([126, 255, 200])

    mask_blue = cv2.inRange(imgHsv, lower_blue, upper_blue)
    return mask_blue


def getContour(img, src):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    imageLists = []
    contourPicture = src.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 500 < area:
            print(area)
            cv2.drawContours(contourPicture, cnt, -1, (0, 255, 0), 2)
            peri = cv2.arcLength(cnt, True)

            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)

            # cv2.rectangle(imgContour, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 1)
            newY = y - 10
            newX = x - 10
            endY = y + h + 10
            endX = x + w + 10

            if newY < 0:
                newY = 0
            if newX < 0:
                newX = 0

            cropped = src[newY:endY, newX:endX]
            height, width, _ = cropped.shape
            r1 = height / float(width)
            r2 = width / float(height)

            # avoid too long rectangles or lines
            if r1 < 1.7 and r2 < 1.7:
                imageLists.append(cropped)
                cv2.rectangle(contourPicture, (newX, newY), (endX, endY), (0, 255, 0), 1)
    return [imageLists, contourPicture]


def getCroppedImages(img):
    # img = cv2.imread(path)
    resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    imgHsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Detect red color
    mask_red = detectRed(imgHsv)
    # Detect blue color
    mask_blue = detectBlue(imgHsv)
    # Combine filters (or)
    mask = mask_red + mask_blue

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    imgOpen = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("img open", imgOpen)

    imgList = getContour(imgOpen, resized)
    return imgList


def findLargestCropped(img_list):
    max_area = 0
    max_index = 0
    for i in range(len(img_list)):
        height, width, _ = img_list[i].shape
        current_area = height * width
        if current_area > max_area:
            max_area = current_area
            max_index = i
    return img_list[max_index]


def open_manually():
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window

    while True:
        file_path = filedialog.askopenfilename()
        print(file_path)
        if len(file_path) == 0:
            cv2.destroyAllWindows()
            break

        img = cv2.imread(file_path)
        images = getCroppedImages(img)
        # cv2.imshow("img", img)
        cv2.imshow("contour", images[1])
        cv2.waitKey()


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img =cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


def getCalssName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'