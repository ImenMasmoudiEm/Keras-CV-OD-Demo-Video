#I used python version 3.8

#libraries installed

#pip install opencv-python Y
#pip install tensorflow==2.12.0
#pip install keras==2.12.0
#pip install keras-cv==0.5.0 or pip install --upgrade git+https://github.com/keras-team/keras-cv 0.6.4
#pip install numpy==1.22.4 Y


import keras_cv
import numpy as np
import cv2

pretrained_model = keras_cv.models.RetinaNet.from_preset("retinanet_resnet50_pascalvoc", bounding_box_format="xywh")

class_ids = ["Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Dining Table", "Dog", "Horse", "Motorbike", "Person", "Potted Plant", "Sheep", "Sofa", "Train", "Tvmonitor", "Total"]
#                 0           1        2       3        4        5      6      7       8       9           10         11      12         13          14           15          16      17       18        19          20

inference_resizing = keras_cv.layers.Resizing(640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh")
prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
            bounding_box_format="xywh",
            from_logits=True,
            # Decrease the required threshold to make predictions get pruned out
            iou_threshold=0.2,
            # Tune confidence threshold for predictions to pass NMS
            confidence_threshold=0.7)

pretrained_model.prediction_decoder = prediction_decoder
class_mapping = dict(zip(range(len(class_ids)), class_ids))

Videos = ['Timeline 1.mov','People.mp4']
cap = cv2.VideoCapture('C:/Users/Lenovo/Desktop/Demo Videos/'+Videos[1])

# Check if camera opened successfully
if (cap.isOpened() == False):
  print("Error opening video stream or file")
while (cap.isOpened()):
    ret, image = cap.read()
    if ret:
        image = np.array(image)
        image = image[0:640, 0:640, :]
        image_batch = np.resize(image,(1, 640, 640, 3))
        y_pred = pretrained_model.predict(image_batch)
        for i in np.arange(0, y_pred['num_detections'][0]):
            confidence = y_pred['confidence'].numpy()[0, i]
            if confidence > 0.5:
                idx = int(y_pred['classes'].numpy()[0, i])
                if class_ids[idx] == class_ids[14]:
                    cow_box = y_pred['boxes'].numpy()[0, i, :]
                    (startX, startY, W, H) = cow_box.astype("int")
                    (startX, startY, endX, endY) = (startX, startY, startX + W, startY + H)
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 5)
        cv2.imshow('Demo',image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else:
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
