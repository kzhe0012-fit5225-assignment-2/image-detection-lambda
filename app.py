from collections import namedtuple
import cv2
import numpy as np
import json
import boto3
from base64 import b64decode

s3 = boto3.client("s3")

yolo_bucket_name = "yolo-file"

confthres = 0.05
nmsthres = 0.1


DetectionResult = namedtuple(
    "DetectionResult", ["item", "accuracy", "x", "y", "width", "height"]
)


def load_yolo():
    weight_yolov3 = s3.get_object(Bucket=yolo_bucket_name, Key="yolov3-tiny.weights")[
        "Body"
    ].read()
    coco_dataset_label = s3.get_object(Bucket=yolo_bucket_name, Key="coco.names")[
        "Body"
    ].read()
    conf_yolov3 = s3.get_object(Bucket=yolo_bucket_name, Key="yolov3-tiny.cfg")[
        "Body"
    ].read()

    labels_coconames = coco_dataset_label.decode("utf8").strip().split("\n")
    return weight_yolov3, conf_yolov3, labels_coconames


def perform_detection(event):
    weights, config, labels = load_yolo()

    b64img = event["image"]
    nparr = np.fromstring(b64decode(b64img), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  #
    (H, W) = img.shape[:2]

    load_yolov3 = cv2.dnn.readNetFromDarknet(config, weights)

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    load_yolov3.setInput(blob)
    outInfo = load_yolov3.getUnconnectedOutLayersNames()

    layerOutputs = load_yolov3.forward(outInfo)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confthres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres, nmsthres)

    results = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            result = DetectionResult(
                labels[classIDs[i]],
                confidences[i],
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
            )
            results.append(result)
            print(result)

    print(results)
    return results


def handler(event, context):
    return {
        "statusCode": 200,
        "body": json.dumps([r._asdict() for r in perform_detection(event)]),
    }
