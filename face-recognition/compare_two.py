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
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./sample-images/dbg.jpg', face)
        face = np.dstack((face, np.fliplr(face)))
        face = face.transpose((2, 0, 1))
        face = face[:, np.newaxis, :, :]
        face = face.astype(np.float32, copy=False)
        face -= 127.5
        face /= 127.5
        # Getting the features from the model
        data = torch.from_numpy(face)
        data = data.to(torch.device("cuda"))
        output = model(data)
        output = output.data.cpu().numpy()
        feature = np.hstack((output[0], output[1]))
        features.append(feature)
    return cosin_metric(features[0], features[1])


if __name__ == '__main__':
    model_path = 'arcface/checkpoints/resnet18_110.pth'
    model = resnet_face18()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = DataParallel(model)
    model.eval()
    print(compare_two(model, ('sample-images/t1.jpg', 'sample-images/ross.jpg')))
    print(compare_two(model, ('sample-images/chandler.png', 'sample-images/ross.jpg')))
    print(compare_two(model, ('sample-images/phoebe.jpg', 'sample-images/ross.jpg')))
