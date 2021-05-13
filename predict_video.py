import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import joblib
from mtcnn_facenet.custom_mtcnn import GetEmb
import cv2

file_path = 'data/train/Trump/test.jpeg'
labels = np.load('label_pair.npy')

getEmbs = GetEmb()
svc = joblib.load('svc_face.pkl')

def _find(frame):

    x = getEmbs.get_embeddings(frame)

    pred_label = svc.predict([x])

    print("PREDICT : ", labels[pred_label])

capture = cv2.VideoCapture(0)

while cv2.waitKey(33) != ord('q'):
    try:
        ret, frame = capture.read()
        img = frame

        faces, _ = getEmbs.mtcnn.detect(frame)# if not FastVersion else [mtcnn(frames), 0]
    # print(faces)

        try:
            for face in faces:
                face = np.trunc(face)
            # print("RECT : {}, {}, {}, {}".format(face[0],face[1],face[2],face[3]))
                lu_x, lu_y, rd_x, rd_y  = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                cv2.rectangle(frame, (lu_x, lu_y), (rd_x, rd_y), (255,0,0), 3)
            #cv2.rectangle(frame, (int(face[0]),int(face[1])), (int(face[2]), int(face[3])), (255,0,0), 3)

        except Exception as e:
            print("Err : ", e)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        _find(im_pil)

        cv2.imshow("face", frame)
    except:
        pass
capture.release()
cv2.destroyAllWindows()

