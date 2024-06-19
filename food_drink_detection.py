import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

class_labels = model.names

food_labels = ['apple', 'banana', 'orange', 'cake', 'sandwich', 'pizza']
drink_labels = ['cup', 'bottle', 'wine glass', 'coffee cup', 'tea cup']

cap = cv2.VideoCapture(0)  

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    results = model(frame)

    detections = results.xyxy[0].cpu().numpy()  

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = class_labels[int(cls)]
        
        if label in food_labels or label in drink_labels:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Food and Drink Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
