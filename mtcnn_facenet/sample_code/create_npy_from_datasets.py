from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
from PIL import Image, ImageDraw
from IPython import display
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(filename):
    img = Image.open(filename)
    img_cropped = mtcnn(img)

    model.classify = True
    img_probs = model(img_cropped.unsqueeze(0))
    return img_probs[0].detach().numpy()



import os
embs = []
labels = []

path = "../data/train"
folder_list = os.listdir(path)
 
for i in folder_list:
    if i == ".DS_Store":
        continue
    _path = path + "/" + i
    file_list = os.listdir(_path)
    
    for _file in file_list:
        if _file == ".DS_Store":
            continue
        print("Embedding : {}".format(_path+"/"+_file))
        embs.append(get_embedding(_path+"/"+_file))
        labels.append(i)

        
e = np.array(embs)
# print(e[0])
np.save('emb.npy', e)
l = np.array(labels)
# print(l[0])
np.save('label.npy', l)

