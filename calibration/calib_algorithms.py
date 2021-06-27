import numpy
import numpy as np
from scipy.special import softmax
from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


class TrainCalibRequest:
    def __init__(self, y_pred, y_target, n_positives, n_negatives):
        self.y_pred = y_pred
        self.y_target = y_target
        self.n_positives = n_positives
        self.n_negatives = n_negatives


def gmm_calib_linear(s, niters=20):
    weights = numpy.array([0.5, 0.5])
    means = numpy.mean(s) + numpy.std(s) * numpy.array([-1, 1])
    var = numpy.var(s)
    threshold = numpy.inf
    for _ in range(niters):
        lls = numpy.log(weights) - 0.5 * numpy.log(var) - 0.5 * (s[:, numpy.newaxis] - means) ** 2 / var
        gammas = softmax(lls, axis=1)
        cnts = numpy.sum(gammas, axis=0)
        weights = cnts / cnts.sum()
        means = s.dot(gammas) / cnts
        var = ((s ** 2).dot(gammas) / cnts - means ** 2).dot(weights)
        threshold = -0.5 * (numpy.log(weights ** 2 / var) - means ** 2 / var).dot([1, -1]) / (means / var).dot([1, -1])
    return threshold, lls[:, means.argmax()] - lls[:, means.argmin()]


def _log_values(points, min_power):
    return 10 ** (numpy.arange(1 - points, 1) / int(points / (-min_power)))


def __meaningful_thresholds(imposters, targets, points, min_far, is_sorted):
    half_points = points // 2

    neg = imposters if is_sorted else numpy.sort(imposters)
    pos = targets if is_sorted else numpy.sort(targets)

    frr_list = _log_values(half_points, min_far)
    far_list = _log_values(points - half_points, min_far)

    t = numpy.zeros((points,))
    t[:half_points] = [__frr_threshold(neg, pos, k, True) for k in frr_list]
    t[half_points:] = [__far_threshold(neg, pos, k, True) for k in far_list]

    t.sort()

    return t


def __far_threshold(imposters, targets, far_value=0.001, is_sorted=False):
    if far_value < 0.0 or far_value > 1.0:
        raise RuntimeError("`far_value' must be in the interval [0.,1.]")

    if len(imposters) < 2:
        raise RuntimeError("the number of negative scores must be at least 2")

    epsilon = numpy.finfo(numpy.float64).eps
    scores = imposters if is_sorted else numpy.sort(imposters)

    if far_value >= (1 - epsilon):
        return numpy.nextafter(scores[0], scores[0] - 1)

    scores = numpy.flip(scores)

    total_count = len(scores)
    current_position = 0

    valid_threshold = numpy.nextafter(
        scores[current_position], scores[current_position] + 1
    )
    current_threshold = 0.0

    while current_position < total_count:

        current_threshold = scores[current_position]
        while (
                current_position < (total_count - 1)
                and scores[current_position + 1] == current_threshold
        ):
            current_position += 1

        future_far = (current_position + 1) / total_count
        if future_far > far_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold


def __frr_threshold(imposters, targets, frr_value=0.001, is_sorted=False):
    epsilon = numpy.finfo(numpy.float64).eps
    scores = targets if is_sorted else numpy.sort(targets)

    if frr_value >= (1 - epsilon):
        return numpy.nextafter(scores[-1], scores[-1] + 1)

    total_count = len(scores)
    current_position = 0

    valid_threshold = numpy.nextafter(
        scores[current_position], scores[current_position] + 1
    )
    current_threshold = 0.0

    while current_position < total_count:

        current_threshold = scores[current_position]

        while (
                current_position < (total_count - 1)
                and scores[current_position + 1] == current_threshold
        ):
            current_position += 1

        future_frr = current_position / total_count
        if future_frr > frr_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold


def ppndf(p):
    """Returns the Deviate Scale equivalent of a false rejection/acceptance ratio
    """

    epsilon = numpy.finfo(numpy.float64).eps
    p_new = numpy.copy(p)
    p_new = numpy.where(p_new >= 1.0, 1.0 - epsilon, p_new)
    p_new = numpy.where(p_new <= 0.0, epsilon, p_new)

    q = p_new - 0.5
    abs_q_smaller = numpy.abs(q) <= 0.42
    abs_q_bigger = ~abs_q_smaller

    retval = numpy.zeros_like(p_new)

    # first part q<=0.42
    q1 = q[abs_q_smaller]
    r = numpy.square(q1)
    opt1 = (
            q1
            * (
                    ((-25.4410604963 * r + 41.3911977353) * r + -18.6150006252) * r
                    + 2.5066282388
            )
            / (
                    (
                            ((3.1308290983 * r + -21.0622410182) * r + 23.0833674374) * r
                            + -8.4735109309
                    )
                    * r
                    + 1.0
            )
    )
    retval[abs_q_smaller] = opt1

    # second part q>0.42
    q2 = q[abs_q_bigger]
    r = p_new[abs_q_bigger]
    r[q2 > 0] = 1 - r[q2 > 0]
    if (r <= 0).any():
        raise RuntimeError("measure::ppndf(): r <= 0.0!")

    r = numpy.sqrt(-1 * numpy.log(r))
    opt2 = (((2.3212127685 * r + 4.8501412713) * r + -2.2979647913) * r + -2.7871893113) / (
            (1.6370678189 * r + 3.5438892476) * r + 1.0)
    opt2[q2 < 0] *= -1
    retval[abs_q_bigger] = opt2

    return retval


def __farfrr(negatives, positives, threshold):
    return (negatives >= threshold).sum() / len(negatives), (positives < threshold).sum() / len(positives)


def det(negatives, positives, n_points, min_far=-8):
    return ppndf(roc(negatives, positives, n_points, min_far))


def roc(negatives, positives, n_points, min_far=-8):
    t = __meaningful_thresholds(negatives, positives, n_points, min_far, False)
    curve = numpy.empty((2, len(t)))
    for i, k in enumerate(t):
        curve[:, i] = __farfrr(negatives, positives, k)
    return curve


def perf_measure(actual, score, treshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    size = len(score)
    for i in range(size):
        predArr = pred_approx_val(score[i], treshold)
        for j in range(len(score[i])):
            if (predArr[j] != actual[i][j] and predArr[j] == 1):
                FP += 1
            if (predArr[j] == actual[i][j] == 1):
                TP += 1
            if (predArr[j] != actual[i][j] and predArr[j] == 0):
                FN += 1
            if (predArr[j] == actual[i][j] == 0):
                TN += 1
    return TP, FP, TN, FN


def pred_approx_val(arr, treshold):
    array_np = np.copy(arr)
    low_val_indices = arr < treshold
    high_val_indices = arr >= treshold
    array_np[low_val_indices] = 0
    array_np[high_val_indices] = 1
    return array_np


def calc_far_frr(FP, FN, totalP, totalN):
    FAR = FP / float(totalP)
    FRR = FN / float(totalN)
    return FAR, FRR


def prepare_graph_far_frr(actual, score, totalP, totalN):
    far = np.zeros(100)
    frr = np.zeros(100)

    for i in range(100):
        _, FP, _, FN = perf_measure(actual, score, i / float(100))
        far[i], frr[i] = calc_far_frr(FP, FN, totalP, totalN)
    return far, frr


def fa_at_fr(request: TrainCalibRequest, fr=10):
    fa_points, fr_points = prepare_graph_far_frr(request.y_target, request.y_pred, request.n_positives,
                                                 request.n_negatives)
    threshold = numpy.argmin(fr_points <= (fr / 100.))

    return 100.0 * fa_points[threshold], threshold

def find_nearest(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return array[idx]

def train_hist_analyze(requests: List[TrainCalibRequest]):
    closest_bins = []

    for request in requests:
        fa, th = fa_at_fr(request)

        values = numpy.histogram(request.y_pred, 100)[0]
        closest_bins.append(find_nearest(values, th))

    return numpy.mean(numpy.array(closest_bins))


def train_linear_classifier(requests: List[TrainCalibRequest]):
    regressor = LinearRegression()

    X = []
    Y = []

    points_of_interest = numpy.array([0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    for request in requests:
        for point in points_of_interest:
            fa, th = fa_at_fr(request, point)

            values = numpy.histogram(request.y_pred, 100)[0]
            X.append(find_nearest(values, th))
            Y.append(fa)

    regressor.fit(np.array(X).reshape((-1, 1)), np.array(Y).reshape((-1, 1)))
    return regressor



def train_clustering(request: TrainCalibRequest, cluster_clazz=DBSCAN, **cluster_params):
    cluster_alg = cluster_clazz(**cluster_params)
    cluster_alg.fit(request.y_target)

    return cluster_alg