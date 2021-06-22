from retinaface import RetinaFace
import cv2
import numpy as np
from skimage import transform as trans

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src = arcface_src
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112):
    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


target_size = 400
max_size = 800
detector = RetinaFace(quality='normal')


def get_faces(img_path):
    im = cv2.imread(img_path)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    predict_res = detector.predict(im, threshold=0.5)
    bbox = np.empty(shape=(len(predict_res), 4))
    landmark = np.empty(shape=(len(predict_res), 5, 2))

    for i, res in enumerate(predict_res):
        bbox[i] = np.array(list(res[k] for k in ('x1', 'y1', 'x2', 'y2') if k in res))
        landmark[i] = np.array(list(res[k] for k in ('left_eye', 'right_eye', 'nose', 'left_lip', 'right_lip') if k in res))
    print(im.shape, bbox.shape, landmark.shape)
    nrof_faces = bbox.shape[0]
    if nrof_faces > 0:
        det = bbox[:, 0:4]
        img_size = np.asarray(im.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                           det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared *
                               2.0)  # some extra weight on the centering
        #_bbox = bounding_boxes[bindex, 0:4]
        _landmark = landmark[bindex]
        warped = norm_crop(im, landmark=_landmark)
        return warped
    else:
        return None


if __name__ == '__main__':
    im1 = './sample-images/t1.jpg'
    face = get_faces(im1)
    cv2.imwrite('./sample-images/align-out.jpg', face)
