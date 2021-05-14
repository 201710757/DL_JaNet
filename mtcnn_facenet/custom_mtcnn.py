from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
from PIL import Image, ImageDraw
from IPython import display

class GetEmb:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def get_embedding(self, filename):
        img = Image.open(filename)
        img_cropped = self.mtcnn(img)

        self.model.classify = True
        embs = self.model(img_cropped.unsqueeze(0))
        return embs[0].detach().numpy()

    def get_embeddings(self, img):
        img_cropped = self.mtcnn(img)

        self.model.classify = True
        embs = self.model(img_cropped.unsqueeze(0))
        return embs[0].detach().numpy()






