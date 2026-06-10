import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

dataset_path = "dataset/leapGestRecog"

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for gesture in os.listdir(person_path):
        gesture_path = os.path.join(person_path, gesture)

        for img in os.listdir(gesture_path):

            img_path = os.path.join(gesture_path, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image,(64,64))

            data.append(image.flatten())
            labels.append(gesture)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = SVC()

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

from sklearn.metrics import classification_report

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, pred))
