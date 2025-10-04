# 🩺 Pneumonia Detection using CNN

Projekt polega na klasyfikacji zdjęć rentgenowskich płuc w celu wykrycia zapalenia płuc (PNEUMONIA) lub obrazu prawidłowego (NORMAL) z wykorzystaniem sieci konwolucyjnych (CNN).

## 📊 Dataset

Dane pochodzą z Kaggle: Chest X-Ray Pneumonia Dataset

# Zbiór zawiera foldery:

📂 train/ – dane treningowe

📂 val/ – dane walidacyjne

📂 test/ – dane testowe

## 🧠 Model

## Model został zbudowany w TensorFlow/Keras. Architektura:

🔲 Conv2D(32) + MaxPooling2D

🔲 Conv2D(64) + MaxPooling2D

🔲 Conv2D(128) + MaxPooling2D

🔲 Flatten

🔲 Dense(64, relu)

🔲 Dense(1, sigmoid)

🛠️ Augmentacja danych

## Dane treningowe są powiększane o:

🔄 rotacje

↔️ przesunięcia

🔍 zoom

🪞 odbicia poziome

⚖️ normalizację pikseli

🚀 Trening

📉 Funkcja straty: binary_crossentropy

⚡ Optymalizator: adam

📈 Metryka: accuracy

🔂 Liczba epok: 10

✅ Wyniki

## Na zbiorze testowym model osiąga dokładność w okolicach 90%+.

🖥️ Technologie

🐍 Python 3

🔬 TensorFlow / Keras

📥 kagglehub (do pobrania danych)

📊 Matplotlib, NumPy

⚙️ Uruchomienie

## 📥 Pobierz dane z Kaggle:

import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")


## ▶️ Uruchom skrypt:

python chest_xray_cnn.py


## 📊 Wyniki treningu pojawią się w konsoli.
