🎯 Real-Time Object Detection and Tracking (Xbox Controller)
A lightweight, real-time object detection and tracking system built using a custom Convolutional Neural Network (CNN) in TensorFlow/Keras and OpenCV — optimized for CPU-only environments.

🚀 Features
🎮 Detects and tracks Xbox controllers in real time

⚡ Runs smoothly on standard laptop CPU — no GPU required

📦 Custom CNN with 4 convolutional blocks

📹 Webcam-based inference using OpenCV

🎯 Integrated CSRT tracker for smooth temporal consistency

🧠 Model Overview
Architecture: 4 Conv2D blocks + GlobalAveragePooling

Outputs:

bbox: normalized bounding-box coordinates

cls: class confidence score

Loss Functions:

MSE for bounding boxes

Binary Crossentropy for classification

Training:

10 epochs on CPU

Batch size: 8

Dataset: 550 Xbox controller images from Roboflow

📂 Project Structure
bash
Copy
Edit
real_time_object_detection/
│
├── src/
│   ├── train.py         # Model architecture and training script
│   ├── detect.py        # Real-time inference from webcam
│   ├── utils.py         # Data loader and helper functions
│
├── models/
│   └── ckpt_last.weights.h5  # Trained model weights
│
├── data/
│   ├── images/          # Training & validation images
│   └── labels/          # YOLO-style annotations
│
├── README.md
└── .gitignore
📈 Results
Metric	Value
Mean IoU	0.75
Confidence ≥ 0.5	0.82
Recall	0.78
Inference Speed	~12.4 FPS (detection only), ~8.6 FPS (with tracking)

⚙️ Requirements
Python 3.10+

TensorFlow

OpenCV

NumPy

Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
📡 Inference (Real-Time)
bash
Copy
Edit
python src/detect.py
Ensure your webcam is accessible and model weights are in models/

📌 Limitations
Only trained for Xbox controllers

May struggle with occlusion, blur, or sharp angles

Static input size (416x416) can introduce distortion

💡 Future Improvements
Multi-class detection support

Efficient backbones (e.g., MobileNetV3)

Raspberry Pi / edge-device deployment

Online learning for adapting to new environments

📄 License
This project is open source and free to use for educational purposes.

🧑‍💻 Author
Shubham Gupta
🎓 B.S. in Computer Science, University of Idaho
