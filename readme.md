# Emotion Detector using ANN (TensorFlow + OpenCV)

This project is a simple emotion detector built during the AI/ML Workshop.  
It detects emotions (Happy, Sad, Neutral, Angry, Surprise, Fear, Disgust) from real photos using a trained ANN model.

---

## 📂 Project Structure

![image](https://github.com/user-attachments/assets/69988744-6e06-4359-8153-8ca76c4bd108)


---

## ⚙ Installation

1. Clone or download this project folder.
2. Install required libraries:

bash
pip install -r requirements.txt


---

## 🚀 How to Train the Model

Make sure you have a proper emotions.csv file inside the data/ folder.

Then run:

bash
python main.py train --csv_path data/emotions.csv


✅ This will:
- Train the ANN model
- Save the best model as models/emotion_model.h5
- Plot training vs validation accuracy


![image](https://github.com/user-attachments/assets/41477650-1f56-47fb-b0b8-d475fd9f4703)


---

## 🤖 How to Predict Emotion (Text Output)

If you want to *predict the emotion* from a real photo and *print* the result in the terminal:

bash
python main.py predict --img_path path_to_your_image.jpg


✅ Example Output:


Predicted Emotion: Happy


---

## 🖼 How to Predict and Show Image (Visual Output)

If you want to *show the image with bounding box + predicted emotion label*:

bash
python main.py predict_visual --img_path path_to_your_image.jpg


✅ This will:
- Detect face(s)
- Predict emotion(s)
- Show the image with labels drawn

Press any key to close the window after viewing.

---

## 📋 Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 📢 Notes

- Input images must have *clear visible faces*.
- Prefer *frontal face images* for better detection.
- Image will automatically resize for easier display.
- Training and prediction assumes grayscale 48x48 images internally.

---

## 🧠 Credits

- TensorFlow / Keras
- OpenCV
- Matplotlib
- Scikit-learn

## Image paths

Built with ❤ during AI/ML Workshop, MCE.

---
