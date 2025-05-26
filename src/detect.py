import cv2
import numpy as np
import tensorflow as tf
from src.train import build_model

MODEL_PATH = "models/ckpt_last.weights.h5"
CONF_THRESH = 0.1

model = build_model()
model.load_weights(MODEL_PATH)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (416, 416)).astype(np.float32) / 255.0

    pred_bbox, pred_cls = model.predict(np.expand_dims(img_resized, 0))
    cls_conf = float(pred_cls[0][0])
    nb = pred_bbox[0]
    x1 = int(nb[0] * w)
    y1 = int(nb[1] * h)
    x2 = int(nb[2] * w)
    y2 = int(nb[3] * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if cls_conf > CONF_THRESH:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"Xbox: {cls_conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()