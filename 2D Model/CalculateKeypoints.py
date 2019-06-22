import os
import cv2


import numpy as np


dir_path = "output/OAS30022/OAS30022_MR_d1324/anat3"
images = os.listdir(dir_path)
for image in images:

    img = cv2.imread(os.path.join(dir_path,image), 0)
    # img = np.uint8(img)
    img = np.asarray(img, dtype=np.uint8)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = np.uint8(img)
    sift = cv2.xfeatures2d.SIFT_create(300, sigma=2.6)
    kp, des = sift.detectAndCompute(img, None)
    print(len(des))



