import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import joblib
from mtcnn_facenet.custom_mtcnn import GetEmb
import cv2


getEmbs = GetEmb()
capture = cv2.VideoCapture(0)

cnt = 0
save_path = 'saveImage/'
name = "kimjihoon"
margin = 60

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
            tmp[0] = int(tmp[0]) - margin
            tmp[1] = int(int(tmp[1]) - margin*1.8)
            tmp[2] = int(tmp[2]) + margin
            tmp[3] = int(tmp[3]) + margin


            draw.rectangle(tmp, outline=(255, 0, 0), width=6)
            cv2.imshow("VideoFrame", cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))

            if cv2.waitKey(33) == ord('c'):
                cnt += 1
                cv2.imwrite(save_path + name + str(cnt) + ".jpg", img[tmp[1]:tmp[3],tmp[0]:tmp[2]])
                print("{}] SAVED !! ".format(cnt))
    except Exception as e:
        print(e)
        pass



#    if cv2.waitKey(33) == ord('u'):
#        margin += 1
#        print("MARGIN : ", margin)
#    elif cv2.waitKey(33) == ord('d'):
#        margin -= 1
#        print("MARGIN : ", margin)

capture.release()
cv2.destroyAllWindows()

