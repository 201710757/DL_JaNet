import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import joblib
from mtcnn_facenet.custom_mtcnn import GetEmb
import cv2

file_path = 'data/train/Trump/test.jpeg'
labels = np.load('savefiles/label_pair.npy')

getEmbs = GetEmb()
svc = joblib.load('savefiles/svc_face.pkl')

def _find(frame):

    x = getEmbs.get_embeddings(frame)

    pred_label = svc.predict([x])
    
    return str(labels[pred_label])
    # print("PREDICT : ", labels[pred_label])

capture = cv2.VideoCapture(0)

while cv2.waitKey(33) != ord('q'):
    ret, frame = capture.read()
    cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame
    frame = Image.fromarray(cap)

    lb = ""
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        lb = _find(im_pil)
        print("PREDICT : ", lb)
    except Exception as e:
        print("Erro : ", e)
        pass


    boxes, _ = getEmbs.mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    try:
        for box in boxes:
            tmp = box.tolist()
            
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20) 
            draw.text((tmp[0],tmp[1]-25), str(lb), font=font, fill=(255,255,255), )
            draw.rectangle(tmp, outline=(255, 0, 0), width=6)
    except Exception as e:
        print(e)
        pass

    n_img = np.array(frame_draw)

    cv2.imshow("VideoFrame", cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))

capture.release()
cv2.destroyAllWindows()
