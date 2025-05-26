import argparse
import os
import random
import numpy as np
import tensorflow as tf
from src.utils import load_data

def build_model():
    inp = tf.keras.Input(shape=(416, 416, 3))
    x = inp
    for f in [16, 32, 64]:
        x = tf.keras.layers.Conv2D(f, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    bbox = tf.keras.layers.Dense(4, activation='sigmoid', name='bbox')(x)
    cls  = tf.keras.layers.Dense(1, activation='sigmoid', name='cls')(x)
    return tf.keras.Model(inp, [bbox, cls])

def data_gen(img_dir, label_dir, batch_size):
    gen = load_data(img_dir, label_dir, img_size=(416, 416))
    while True:
        imgs, bboxes, clss = [], [], []
        for _ in range(batch_size):
            try:
                img, boxes, _ = next(gen)
            except StopIteration:
                gen = load_data(img_dir, label_dir, img_size=(416, 416))
                img, boxes, _ = next(gen)
            if boxes.shape[0] > 0 and random.random() < 0.5:
                img = np.fliplr(img)
                x1, y1, x2, y2 = boxes[0]
                img_w = img.shape[1]
                boxes[0] = [img_w - x2, y1, img_w - x1, y2]
            imgs.append(img)
            if boxes.shape[0] > 0:
                b = boxes[0] / np.array([416, 416, 416, 416], dtype=np.float32)
                c = 1.0
            else:
                b = np.zeros((4,), dtype=np.float32)
                c = 0.0
            bboxes.append(b)
            clss.append(c)
        yield np.stack(imgs), {'bbox': np.stack(bboxes), 'cls': np.array(clss, dtype=np.float32)}

def main(args):
    model = build_model()
    model.compile(optimizer='adam',
                  loss={'bbox': 'mse', 'cls': 'binary_crossentropy'},
                  loss_weights={'bbox': 1.0, 'cls': 1.0})
    steps = max(1, len(os.listdir(args.img_dir)) // args.batch_size)
    model.fit(data_gen(args.img_dir, args.label_dir, args.batch_size),
              steps_per_epoch=steps,
              epochs=args.epochs)
    model.save_weights(os.path.join(args.model_dir, 'ckpt_last.weights.h5'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir',   default='data/images/xbox_controller')
    p.add_argument('--label_dir', default='data/labels/xbox_controller')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--model_dir',  default='models')
    args = p.parse_args()
    main(args)
