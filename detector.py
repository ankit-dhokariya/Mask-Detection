import cv2
import pickle
import argparse
from Conv3x3 import Conv3x3
from MaxPool import MaxPool2
from Logistic import Logistic

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
args = vars(ap.parse_args())

video = cv2.VideoCapture(0)

conv = Conv3x3(8)
pool = MaxPool2()
nn = Logistic()

mask_face = [1, 0]
no_mask = [0, 1]

with open(args["model"], 'rb') as f:
    weights = pickle.load(f)

conv.filters = weights[0]
nn.theta1 = weights[1]
nn.theta2 = weights[2]
nn.theta3 = weights[3]
nn.theta4 = weights[4]

while True:

    ret, frame = video.read()
    if ret:
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        if len(faces):
            for x, y, w, h in faces:
                input = []
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cropped = gray[y: (y + h), x: (x + w)]
                resized = cv2.resize(cropped, (30, 30), interpolation=cv2.INTER_NEAREST)
                input = resized.reshape(1, resized.shape[0], resized.shape[1])
                out = conv.forward((input / 255) - .5)
                out = pool.forward(out)
                out, cost = nn.forward(out, None)
                # print(out)
                if (out == mask_face).all():
                    prediction = 'Mask Face'
                elif (out == no_mask).all():
                    prediction = 'No Mask'
                cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        cv2.imshow('face detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
