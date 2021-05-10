from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open('sample.jpg')

img_cropped = mtcnn(img)

model.classify = True
img_probs = model(img_cropped.unsqueeze(0))
print(img_probs)

emb = []
def get_embedding(filename):
    img = Image.open(file_name)
    img_cropped = mtcnn(img)

    model.classify = True
    img_probs = model(img_cropped.unsqueeze(0))
    emb.append(img_probs)



import os

path = "./train"
folder_list = os.listdir(path)
 
for i in folder_list:
    _path = path + "/" + i
    file_list = os.listdir(_path)
    
    for _file in file_list:
        get_embedding(_path+"/"+_file)





import sys
sys.exit(0)

capture = cv2.VideoCapture(0)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    cap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = Image.fromarray(cap)#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
     
    boxes, _ = mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    try:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    except Exception as e:
        pass
    
    n_img = np.array(frame_draw)

    cv2.imshow("VideoFrame", cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))

capture.release()
cv2.destroyAllWindows()
