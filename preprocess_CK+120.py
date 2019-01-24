# 0=anger, 1=contempt, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise
# contains 45, 18, 59, 25, 69, 28, 83 images

import os
import sys
import numpy as np
import h5py
import cv2


def main():
    ck_path = 'CK+120'
    data_x = []
    data_y = []

    out_path = os.path.join('data', 'CK_data.h5')
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    cat = os.listdir(ck_path)
    cat.sort()
    for label in cat:
        cur_path = os.path.join(ck_path, label)
        items = os.listdir(cur_path)
        items.sort()
        for item in items:
            img = cv2.imread(os.path.join(cur_path, item), 0)
            data_x.append(img.tolist())
            data_y.append(int(label))

    print(np.shape(data_x))
    print(np.shape(data_y))

    datafile = h5py.File(out_path, 'w')
    datafile.create_dataset("data_pixel", dtype='uint8', data=data_x)
    datafile.create_dataset("data_label", dtype='int64', data=data_y)
    datafile.close()

    print("Save data finish!!!")


if __name__ == "__main__":
    sys.exit(main())
