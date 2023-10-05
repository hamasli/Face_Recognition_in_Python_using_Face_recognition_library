import cv2;
import face_recognition;
import os;

def read_img(path):
  img=cv2.imread(path);
  (h,w)=img.shape[:2];
  width=500;
  ratio=width/float(w);
  height=int(h*ratio);
  return cv2.resize(img,(width,height));



known_encodings=[];
known_names=[];
known_dir='known';


for file in os.listdir(known_dir):
  img=read_img(known_dir+'/'+file);
  img_enc=face_recognition.face_encodings(img)[0];
  known_encodings.append(img_enc);
  known_names.append(file.split('.')[0]);

print(known_encodings);


unkown_dir='unknown';


name=[];
for file in os.listdir(unkown_dir):
  print("processing file ",file);
  img=read_img(unkown_dir+'/'+file);
  uimg_enc=face_recognition.face_encodings(img)[0];
  results=face_recognition.compare_faces(known_encodings,uimg_enc);
  # fdistance=face_recognition.face_distance(known_encodings,uimg_enc);
  for j,i in enumerate(range(len(results))):
    if results[i]:
      print(known_names[i]);
      name=known_names[i];
      (top,right,bottom,left)=face_recognition.face_locations(img)[0];
      cv2.rectangle(img,(left,top),(right,bottom),color=(0,255,0),thickness=2);
      cv2.putText(img,name,(left-20,top-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1);
      cv2.imshow(f"image{i}",img);
      cv2.waitKey(0);






