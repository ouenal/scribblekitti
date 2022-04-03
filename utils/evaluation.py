import numpy as np

def compute_iou(cm, ignore_zero=False):
    cm = cm.cpu().detach().numpy()
    if cm.sum() == 0: return 0, 0

    tp = np.diag(cm)
    with np.errstate(divide='ignore'):
        ciou = tp / (cm.sum(1) + cm.sum(0) - tp)
    if ignore_zero:
        ciou = ciou[1:]
    miou = np.nanmean(ciou) * 100
    return ciou, miou