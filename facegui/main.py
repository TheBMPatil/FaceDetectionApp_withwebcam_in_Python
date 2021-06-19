from tkinter import *
from tkinter import messagebox, Label, Button
import face_recognition
from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np

top = Tk()
# size
top.title('Face Detection App')
top.geometry("500x400")
selimg: Label = Label(top,
                      text="Click the Buttons to detect Faces",
                      font=("Eras Demi ITC", "20")).place(x=50, y=50)


def detectface():
    messagebox.showinfo("DetectFace", message="Click ok to Proceed!")
    test_image = face_recognition.load_image_file('./face_data/test1.jpg')
    face_locations = face_recognition.face_locations(test_image)
    pil_image = Image.fromarray(test_image)
    draw = ImageDraw.Draw(pil_image)
    for (t, r, b, l), face_encoding in zip(face_locations, test_image):
        draw.rectangle(((l, t), (r, b)), outline=(0, 750, 0), width=3)
    del draw
    pil_image.show()


def detectfacelive():
    messagebox.showinfo("DetectFace", message="Click ok to Proceed!")
    video_capture = cv2.VideoCapture(0)
    bhagvat_img = face_recognition.load_image_file("face_data/bhagvat.jpg")
    BM_face_encoding = face_recognition.face_encodings(bhagvat_img)[0]
    sid_image = face_recognition.load_image_file("face_data/sid.jpg")
    sid_face_encoding = face_recognition.face_encodings(sid_image)[0]
    known_face_encodings = [
        BM_face_encoding,
        sid_face_encoding
    ]
    known_face_names = [
        "Bhagvat",
        "Sid"
    ]
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


findface: Button = Button(top,
                          text="Detect faces in image"
                          , command=detectface).place(x=100, y=100)
findfacelive: Button = Button(top,
                              text="Detect faces live using Camera"
                              , command=detectfacelive).place(x=250, y=100)
top.mainloop()
