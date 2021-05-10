from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

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
