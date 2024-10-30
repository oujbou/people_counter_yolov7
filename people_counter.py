import cv2
import torch
import numpy as np


model = torch.hub.load('yolov7-main/', 'custom', 'yolov7.pt', source='local')
model.eval()

cap = cv2.VideoCapture("video_sample.mp4")

people_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]

    people_detections = detections[detections['name'] == 'person']
    people_count = len(people_detections)

    for _, detection in people_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        confidence = detection['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {int(confidence * 100)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("People Counting with YOLOv7", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
