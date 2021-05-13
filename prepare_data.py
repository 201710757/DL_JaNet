import os
from mtcnn_facenet.custom_mtcnn import GetEmb
import numpy as np
embs = []
labels = []

path = "./data/train"
getEmbs = GetEmb()

folder_list = os.listdir(path)
for folder in folder_list:
    if folder == ".DS_Store":
        continue
    _path = path + '/' + folder
    file_list = os.listdir(_path)

    for _file in file_list:
        if _file == ".DS_Store":
            continue
        print("Embedding : {}".format(_path + '/' + _file))
        embs.append(getEmbs.get_embedding(str(_path + "/" + _file)))
        labels.append(folder)

e = np.array(embs)
np.save('emb.npy', e)
l = np.array(labels)
np.save('label.npy', l)










