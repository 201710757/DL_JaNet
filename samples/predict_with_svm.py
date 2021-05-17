import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import joblib
from mtcnn_facenet.custom_mtcnn import GetEmb

file_path = 'data/train/Trump/test.jpeg'
labels = np.load('label_pair.npy')

getEmbs = GetEmb()
svc = joblib.load('svc_face.pkl')
x = getEmbs.get_embedding(file_path)

pred_label = svc.predict([x])

print("PREDICT : ", labels[pred_label][0])

