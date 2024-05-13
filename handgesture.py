import cv2
import numpy as np

# parameters
width, height = 640, 480
threshold = 0.8  # minimum probability to filter weak detections

# load model
net = cv2.dnn.readNetFromTensorflow("hand_frozen_graph.pb")

# capture video
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    # read frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # flip the frame horizontally

    # prepare blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False)

    # set input and forward pass
    net.setInput(blob)
    out = net.forward()

    # get detection
    detection = out[0, 0, :, :]
    for i in range(detection.shape[0]):
        confidence = detection[i, 2]
        if confidence > threshold:
            class_id = detection[i, 1]
            # get coordinates
            x1 = int(detection[i, 3] * width)
            y1 = int(detection[i, 4] * height)
            x2 = int(detection[i, 5] * width)
            y2 = int(detection[i, 6] * height)
            # draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

    # show output
    cv2.imshow("Hand Gesture Recognition", frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
