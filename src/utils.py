import os
import glob
import cv2
import numpy as np

def load_data(img_dir, label_dir, img_size=(416, 416)):
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        base = os.path.basename(img_path)
        label_path = os.path.join(label_dir, os.path.splitext(base)[0] + '.txt')
        boxes, class_ids = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls_id, x_c, y_c, w, h = map(float, line.split())
                    img_w, img_h = img_size
                    x_c *= img_w; y_c *= img_h
                    w   *= img_w; h   *= img_h
                    x1 = x_c - w/2; y1 = y_c - h/2
                    x2 = x_c + w/2; y2 = y_c + h/2
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(int(cls_id))
        yield img, np.array(boxes, dtype=np.float32), np.array(class_ids, dtype=np.int32)