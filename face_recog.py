import cv2
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file("D:\z\elon.jpg")
imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)

imgelon_test = face_recognition.load_image_file("D:\z\kalam.jpg")
imgelon_test = cv2.cvtColor(imgelon_test,cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(imgelon)[0]
encodeelon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)

face_loc_test = face_recognition.face_locations(imgelon_test)[0]
encodeelon_test = face_recognition.face_encodings(imgelon_test)[0]
cv2.rectangle(imgelon_test,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeelon],encodeelon_test)
faceDis = face_recognition.face_distance([encodeelon],encodeelon_test)
print(result,faceDis)
cv2.putText(imgelon_test,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('ELON MUSK',imgelon)
cv2.imshow('ELON MUSK_TEST',imgelon_test)

cv2.waitKey(0)