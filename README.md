# Hand Gesture Recognition ✋🤖

A real-time hand gesture recognition system using **MediaPipe**, **CNN (TensorFlow/Keras)**, and **OpenCV**.  
The project detects a hand from webcam input and classifies static gestures with high accuracy.

---

## Features
- Real-time gesture recognition via webcam
- MediaPipe-based hand detection
- CNN-based gesture classification
- Robust to finger bends and small rotations
- Custom dataset collection & training
- Reproducible environment using requirements.txt

---

## Supported Gestures
- Fist ✊
- Palm ✋
- Peace ✌️
- OK 👌

---

## Tech Stack
- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn

---

## Project Structure
hand_gesture/
├── images/ # Dataset
│ ├── fist/
│ ├── palm/
│ ├── peace/
│ |── ok/
| └── rock/
├── models/ # Trained CNN model
├── scripts/
│ ├── train_cnn.py
│ ├── live_cnn_predict.py
│ └── hand_test.py
├── requirements.txt
└── README.md


---

## Setup (Windows)

```bash
git clone https://github.com/Nikk118/hand-gesture-recognition.git
cd hand-gesture-recognition
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
