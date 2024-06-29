import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from datetime import datetime
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


path = "D:\image_attendance"
images = []
class_names = []
mylist = os.listdir(path)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    class_names.append(os.path.splitext(cl)[0])
    
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def findEncoding(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        my_data_list = f.readlines()
        name_list = []
        print(my_data_list)
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')



encode_list_known = findEncoding(images)
print("encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img,(0,0),None,0.25,0.25)
    img_small = cv2.cvtColor(img_small,cv2.COLOR_BGR2RGB)

    faces_cur_frame = face_recognition.face_locations(img_small)
    encodes_cur_frame = face_recognition.face_encodings(img_small,faces_cur_frame)

    if len(encodes_cur_frame) == 0:
        cv2.putText(img, "No face found", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    else:
        for encodeface,faceloc in zip(encodes_cur_frame,faces_cur_frame):
            matches = face_recognition.compare_faces(encode_list_known,encodeface)
            face_dis = face_recognition.face_distance(encode_list_known,encodeface)
            print(face_dis)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = class_names[match_index].upper()
                #print(name)
                y1,x2,y2,x1 = faceloc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)
                speak(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)


#face_loc = face_recognition.face_locations(imgelon)[0]
#encodeelon = face_recognition.face_encodings(imgelon)[0]
#cv2.rectangle(imgelon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)

#face_loc_test = face_recognition.face_locations(imgelon_test)[0]
#encodeelon_test = face_recognition.face_encodings(imgelon_test)[0]
#cv2.rectangle(imgelon_test,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(255,0,255),2)

#result = face_recognition.compare_faces([encodeelon],encodeelon_test)
#faceDis = face_recognition.face_distance([encodeelon],encodeelon_test)
