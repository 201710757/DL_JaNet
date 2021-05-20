import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import joblib
from mtcnn_facenet.custom_mtcnn import GetEmb
import cv2

labels = np.load('label_pair.npy')
credit = np.zeros(len(labels))


getEmbs = GetEmb()
svc = joblib.load('svc_face.pkl')

def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def _find(frame):

    x = getEmbs.get_embeddings(frame)

    pred_label = svc.predict([x])
    
    credit[pred_label] += 1

    return str(labels[pred_label])
    # print("PREDICT : ", labels[pred_label])

capture = cv2.VideoCapture(0)

while cv2.waitKey(33) != ord('q'):
    ret, frame = capture.read()
    cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame
    frame = Image.fromarray(cap)

    lb = ""

    boxes, _ = getEmbs.mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    try:
        for box in boxes:
            tmp = box.tolist()
            #print(tmp[0], tmp[1], tmp[2], tmp[3])
            try:
                img_ = img[int(tmp[1])-1:int(tmp[3])+1,int(tmp[0])-1:int(tmp[2])+1]
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img_)
                lb = _find(im_pil)
                print("PREDICT : ", lb)
            except Exception as e:
                #print("Erro : ", e)
                pass


            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
            draw.text((tmp[0],tmp[1]-25), str(lb), font=font, fill=(255,255,255), )
            draw.rectangle(tmp, outline=(255, 0, 0), width=6)
    except Exception as e:
        print(e)
        pass

    n_img = np.array(frame_draw)

    cv2.imshow("VideoFrame", cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))
#print("{}. detected - 5000원 결제되었습니다.".format(labels[np.argmax(softmax(credit))]))
capture.release()
cv2.destroyAllWindows()
