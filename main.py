import face_recognition
import cv2
import os
import numpy as np
import time

def open_camera():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        return None
    return cap

def main():
    
    print("starting")
    matt_image = face_recognition.load_image_file("images/matt1.jpg")
    matt_image = cv2.resize(matt_image, (0, 0), fx=0.5, fy=0.5)

    brendan_image = face_recognition.load_image_file("images/brendan1.jpg")
    brendan_image = cv2.resize(brendan_image, (0, 0), fx=0.5, fy=0.5)

    matt_embed = face_recognition.face_encodings(matt_image)[0]
    brendan_embed = face_recognition.face_encodings(brendan_image)[0]


    known_embeddings = [matt_embed, brendan_embed]
    known_face_names = ["Matthew", "Brendan"]

    frame_count = 0

    cap = open_camera()
    while True:

        if cap is None:
            print("cap unopened")
            time.sleep(1)
            cap = open_camera()
            continue


        frame_count += 1
        if frame_count % 5 != 0:
            continue

        ret, frame = cap.read()

        if not ret:
            print("failed to capture")
            cap.release()
            cap = None
            time.sleep(0.5)
            continue

        
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_embeddings, face_encoding)
            name = "Unknown"

            if True in matches:
                 first_match_index = matches.index(True)
                 name = known_face_names[first_match_index]
            
            face_names.append(name)

        print("Found faces: ", face_names)


    



if __name__ == "__main__":
    main()
