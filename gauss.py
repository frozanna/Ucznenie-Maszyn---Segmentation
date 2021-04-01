import numpy as np


def gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)  # (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  # (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))


def generate_hm(height, width, landmarks, s=5):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
    """
    Nlandmarks = len(landmarks)
    hm = np.zeros((height, width, Nlandmarks), dtype=np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1, -1]):

            hm[:, :, i] = gaussian_k(landmarks[i][0],
                                     landmarks[i][1],
                                     s, height, width)
        else:
            hm[:, :, i] = np.zeros((height, width))
    return hm


def get_y_as_heatmap(df, height, width, sigma=5):

    columns_lmxy = df.columns[:-1]  # the last column contains Image
    columns_lm = []
    for c in columns_lmxy:
        c = c[:-2]
        if c not in columns_lm:
            columns_lm.extend([c])

    y_train = []
    for i in range(df.shape[0]):
        landmarks = []
        for colnm in columns_lm:
            x = df[colnm + "_x"].iloc[i]
            y = df[colnm + "_y"].iloc[i]
            landmarks.append([x, y])

        y_train.append(generate_hm(height, width, landmarks, sigma))
    y_train = np.array(y_train)
    return(y_train, df[columns_lmxy], columns_lmxy)
