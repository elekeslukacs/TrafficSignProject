import numpy as np
import cv2
import pickle
import utils
import tkinter
from tkinter import filedialog

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
pickle_in = open("model1.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)

model2 = pickle.load(open("model_trained2.p", "rb"))


def predictAllImages(img):
    images = utils.getCroppedImages(img)
    cropped_images = images[0]
    cv2.imshow("Original image", images[1])
    for cropped in cropped_images:
        src = np.asarray(cropped)
        src = cv2.resize(src, (32, 32))
        src = utils.preprocessing(src)
        src = src.reshape(1, 32, 32, 1)

        # predict
        predictions = model2.predict(src)
        classIndex = model2.predict_classes(src)
        probability = np.amax(predictions)
        print(str(classIndex) + " " + str(utils.getCalssName(classIndex)))
        print(str(round(probability * 100, 2)) + "%")
        resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
        if probability > threshold:
            cv2.putText(resized, str(classIndex) + " " + str(utils.getCalssName(classIndex)), (8, 25), font, 0.5,
                        (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(resized, str(round(probability * 100, 2)) + "%", (16, 45), font, 0.5, (0, 255, 0), 2,
                        cv2.LINE_AA)

        cv2.imshow("Result", resized)
        cv2.waitKey(0)


def predictImage(img):
    images = utils.getCroppedImages(img)
    cropped_images = images[0]
    cv2.imshow("Original image", images[1])
    if cropped_images:
        largest_sign = utils.findLargestCropped(cropped_images)
        src = np.asarray(largest_sign)
        src = cv2.resize(src, (32, 32))
        src = utils.preprocessing(src)
        src = src.reshape(1, 32, 32, 1)

        # predict
        predictions = model2.predict(src)
        classIndex = model2.predict_classes(src)
        probability = np.amax(predictions)

        print(str(classIndex) + " " + str(utils.getCalssName(classIndex)))
        print(str(round(probability * 100, 2)) + "%")
        resized = cv2.resize(largest_sign, (256, 256), interpolation=cv2.INTER_AREA)

        if probability > threshold:
            # print(str(classIndex) + " " + str(utils.getCalssName(classIndex)))
            # print(str(round(probability * 100, 2)) + "%")

            cv2.putText(resized, str(classIndex) + " " + str(utils.getCalssName(classIndex)), (8, 25), font, 0.5,
                        (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(resized, str(round(probability * 100, 2)) + "%", (16, 45), font, 0.5, (0, 255, 0), 2,
                        cv2.LINE_AA)

            cv2.imshow("Result", resized)

def predictImage2(img):
    images = utils.getCroppedImages(img)
    cropped_images = images[0]
    cv2.imshow("Original image", images[1])
    if cropped_images:
        largest_sign = utils.findLargestCropped(cropped_images)
        src = np.asarray(largest_sign)
        src = cv2.resize(src, (64, 64), interpolation=cv2.INTER_LINEAR)
        # src = utils.preprocessing(src)
        src = src.reshape(1, 64, 64, 3)

        # predict
        predictions = model.predict(src)
        classIndex = model.predict_classes(src)
        probability = np.amax(predictions)

        print(str(classIndex) + " " + str(utils.getCalssName(classIndex)))
        print(str(round(probability * 100, 2)) + "%")
        resized = cv2.resize(largest_sign, (256, 256), interpolation=cv2.INTER_AREA)

        if probability > threshold:
            # print(str(classIndex) + " " + str(utils.getCalssName(classIndex)))
            # print(str(round(probability * 100, 2)) + "%")

            cv2.putText(resized, str(classIndex) + " " + str(utils.getCalssName(classIndex)), (8, 25), font, 0.5,
                        (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(resized, str(round(probability * 100, 2)) + "%", (16, 45), font, 0.5, (0, 255, 0), 2,
                        cv2.LINE_AA)

            cv2.imshow("Result", resized)

def select_pictures():
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    while True:
        file_path = filedialog.askopenfilename()
        print(file_path)
        if len(file_path) == 0:
            cv2.destroyAllWindows()
            break

        img = cv2.imread(file_path)
        predictAllImages(img)


def webcam_mode():
    while True:
        # Read image
        success, img_original = cap.read()

        predictImage2(img_original)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


op = 1
while op != 0:
    print("1 - Open files")
    print("2 - Open camera - Press 'Q' to exit.")
    print("0 - Exit")

    op = int(input("Enter your value: "))

    if op == 1:
        select_pictures()
    elif op == 2:
        webcam_mode()
    elif op == 0:
        break
