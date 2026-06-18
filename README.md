# Face Detection and Recognition with DLib

## Overview

This project implements a real-time face detection and recognition system using DLib, OpenCV, and the Face Recognition library.

The system detects human faces from images or webcam video, extracts facial features using DLib's 128-dimensional face embeddings, and identifies known individuals by comparing facial encodings.

## Features

* Face Detection using DLib HOG Detector
* Facial Landmark Detection (68-Point Model)
* Face Encoding using 128-D Feature Vectors
* Face Recognition using Euclidean Distance Matching
* Real-Time Webcam Recognition
* Multiple Face Support

## Tech Stack

* Python
* OpenCV
* DLib
* Face Recognition Library
* NumPy

## Project Structure

```text
FaceRecognitionDlib/
│
├── data/
│   └── known_faces/
│       ├── person1.jpg
│       ├── person2.jpg
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat
│
├── src/
│   ├── encode_faces.py
│   ├── webcam_recognition.py
│   └── landmark_detection.py
│
├── encodings.pkl
├── README.md
└── pyproject.toml
```

## Working

### Step 1: Face Detection

The DLib frontal face detector identifies faces in an image using Histogram of Oriented Gradients (HOG).

### Step 2: Face Encoding

Each detected face is converted into a 128-dimensional embedding vector.

### Step 3: Face Matching

The generated embeddings are compared against stored embeddings using Euclidean Distance.

### Step 4: Recognition

If the distance is below a threshold, the person is recognized; otherwise, the face is marked as Unknown.

## Facial Landmark Detection

The project also demonstrates DLib's 68-point facial landmark model:

* Jawline
* Eyebrows
* Eyes
* Nose
* Mouth

## Installation

```bash
uv venv --python 3.11
.venv\Scripts\activate

uv add opencv-python
uv add numpy
uv add pillow
uv add dlib
uv add face-recognition
```

## Usage

### Encode Known Faces

```bash
uv run src/encode_faces.py
```

### Real-Time Face Recognition

```bash
uv run src/webcam_recognition.py
```

### Facial Landmark Detection

```bash
uv run src/landmark_detection.py
```

## Output

* Detects faces in real time.
* Identifies known individuals.
* Displays names above detected faces.
* Visualizes 68 facial landmark points.

## Learning Outcomes

* Computer Vision Fundamentals
* Face Detection Techniques
* Facial Landmark Detection
* Feature Extraction
* Face Recognition Systems
* Real-Time Video Processing

## Author

**SHARVESWARAN V**
