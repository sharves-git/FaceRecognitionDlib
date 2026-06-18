import cv2
import face_recognition
import pickle

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

video = cv2.VideoCapture(0)

while True:

    success, frame = video.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)

    encodings = face_recognition.face_encodings(
        rgb,
        locations
    )

    for face_encoding, face_location in zip(encodings, locations):

        distances = face_recognition.face_distance(
            data["encodings"],
            face_encoding
        )

        name = "Unknown"
        
        if len(distances) > 0:
            min_distance = min(distances)
            min_index = list(distances).index(min_distance)
            
            # If distance is below threshold (0.6), recognize the face
            if min_distance < 0.6:
                name = data["names"][min_index]

        top, right, bottom, left = face_location


        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()