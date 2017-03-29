import face_recognition
import glob
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def get_image_filenames ():
    picture_file_extensions = ("*.jpg","*.png")
    picture_filenames = []
    for picture_file_extension in picture_file_extensions:
        picture_filenames.extend(glob.glob('known_faces' + "/" + picture_file_extension))
    return picture_filenames

def load_known_image_encodings (picture_filenames):
    dict_known_image_encodings = {}
    for file_name in picture_filenames:
        print(file_name)
        dict_known_image_encodings[os.path.basename(file_name).split(".")[0]] = create_image_encoding_from_file(file_name)
    return dict_known_image_encodings

def create_image_encoding_from_file (file_name):
    return create_image_encodings(face_recognition.load_image_file(file_name))

def create_image_encodings (image, face_locations=None):
    return face_recognition.face_encodings(image, face_locations)

def identify_persons (unknown_image_encodings, known_image_encodings):
    names = []
    for file_name in known_image_encodings:
        for unknown_image_encoding in unknown_image_encodings:
            result = face_recognition.compare_faces(known_image_encodings[file_name], unknown_image_encoding)
            if result[0] == True:
                names.append(file_name)
                break
    return names

def make_makeup (frame, face_locations):
    face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
    pil_image = Image.fromarray(frame)
    d = ImageDraw.Draw(pil_image, "RGBA")
    for face_landmarks in face_landmarks_list:
        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
  
        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
  
        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
  
        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    return np.array(pil_image)

def show_names(frame, names):
    if len(names) == 0: return frame
    pil_image = Image.fromarray(frame)
    d = ImageDraw.Draw(pil_image, "RGBA")
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 48)
    d.text((100, 100), "Hello, {}".format(names[0]), (255, 255, 255), font=font)
    return np.array(pil_image)

if __name__ == '__main__':
    known_image_encodings = load_known_image_encodings(get_image_filenames())
  
    cap = cv2.VideoCapture(-1)
    process_frame = False
    names = []
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        if process_frame:
            scale_factor = 0.25
            small = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
            face_locations = face_recognition.face_locations(small)
            scaled_locations = []
            for location in face_locations:
                scaled_location = tuple(e*int((1/scale_factor)) for e in location)
                scaled_locations.append(scaled_location)
            if len(face_locations) != 0 :
                unknown_image_encodings = create_image_encodings(frame, scaled_locations)
                names = identify_persons(unknown_image_encodings, known_image_encodings)
                frame = show_names(frame, names)
                # frame = make_makeup(frame, scaled_locations)
                
        if not process_frame and len(names) > 0:
            frame = show_names(frame, names)
            names = []
        cv2.imshow('Test', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        process_frame = not process_frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    