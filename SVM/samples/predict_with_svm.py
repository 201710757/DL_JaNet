import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import joblib

def get_embedding(filename):
    img = Image.open(filename)
    
    img_cropped = mtcnn(img)
    
    model.classify = True
    img_probs = model(img_cropped.unsqueeze(0))
    
    return img_probs[0].detach().numpy()

svc = joblib.load('SVM/svc_face.pkl')
device = 'cpu'
mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval()
x = get_embedding('data/train/LeeYoungji/test.jpg')
# #y = labelData
print("PREDICT : ", svc.predict([x]))

