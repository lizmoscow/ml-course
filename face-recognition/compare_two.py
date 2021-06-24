from prepare_picture import get_faces
import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from arcface.models import *

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def compare_two(model, img_to_compare: tuple):
    features = []
    for img in img_to_compare:
        # Loading and preparing the image
        face = get_faces(img)
        face = np.dstack((face, np.fliplr(face)))
        face = face.transpose((2, 0, 1))
        face = face[:, np.newaxis, :, :]
        face = face.astype(np.float32, copy=False)
        face -= 127.5
        face /= 127.5
        # Getting the features from the model
        data = torch.from_numpy(face)
        data = data.to(torch.device("cpu"))
        output = model(data)
        output = output.data.cpu().numpy()

        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))
        features.append(feature)
    return cosin_metric(features[0], features[1])

if __name__ == '__main__':
    model = resnet18()
    model.load_state_dict(torch.load('model/resnet18.pth'))
    # model = DataParallel(model)
    print(compare_two(model, ('sample-images/t1.jpg', 'sample-images/ross.jpg')))
