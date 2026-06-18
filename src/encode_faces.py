import face_recognition
import os
import pickle

known_encodings = []
known_names = []

dataset_path = "data/known_faces"

for file in os.listdir(dataset_path):

    image_path = os.path.join(dataset_path, file)

    image = face_recognition.load_image_file(image_path)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:

        known_encodings.append(encodings[0])

        name = os.path.splitext(file)[0]

        known_names.append(name)

        print(f"Encoded: {name}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Encodings saved successfully")