import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras import layers, models


# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)



# Ścieżka do folderu z danymi
print("Pliki w folderze:", os.listdir(path))





# Przygotowanie generatora dla zbioru treningowego z augmentacją
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizacja obrazów do zakresu [0,1]
    rotation_range=30,  # Obrót obrazów o losowe kąty
    width_shift_range=0.2,  # Przesunięcie obrazów w poziomie
    height_shift_range=0.2,  # Przesunięcie obrazów w pionie
    shear_range=0.2,  # Losowa zmiana kształtu obrazu
    zoom_range=0.2,  # Losowe skalowanie
    horizontal_flip=True,  # Odbicie poziome
    fill_mode='nearest'  # Wypełnianie pustych miejsc w obrazie
)

# Przygotowanie generatora dla zbioru walidacyjnego i testowego bez augmentacji
test_datagen = ImageDataGenerator(rescale=1./255)

# Załaduj dane z odpowiednich folderów
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(path, "chest_xray/train"),  # Folder z danymi treningowymi
    target_size=(224, 224),  # Zmiana rozmiaru obrazów na 224x224
    batch_size=32,  # Rozmiar batcha
    class_mode='binary'  # Klasyfikacja binarna: NORMAL vs PNEUMONIA
)

validation_generator = test_datagen.flow_from_directory(
    directory=os.path.join(path, "chest_xray/val"),  # Folder z danymi walidacyjnymi
    target_size=(224, 224),  # Zmiana rozmiaru obrazów na 224x224
    batch_size=32,  # Rozmiar batcha
    class_mode='binary'  # Klasyfikacja binarna: NORMAL vs PNEUMONIA
)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(path, "chest_xray/test"),  # Folder z danymi testowymi
    target_size=(224, 224),  # Zmiana rozmiaru obrazów na 224x224
    batch_size=32,  # Rozmiar batcha
    class_mode='binary'  # Klasyfikacja binarna: NORMAL vs PNEUMONIA
)





# Definicja modelu CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 1 jednostka wyjściowa dla klasyfikacji binarnej
])

# Kompilacja modelu
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Można dostosować na podstawie liczby obrazów w zbiorze treningowym
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50  # Można dostosować na podstawie liczby obrazów w zbiorze walidacyjnym
)



# Ocena modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(test_generator, steps=50)  # Można dostosować do liczby obrazów w zbiorze testowym
print(f'Test accuracy: {test_acc}')
