#!/usr/bin/env python
"""
coding=utf-8

Code template courtesy https://github.com/bjherger/Python-starter-repo

"""
import logging

import logging
import os
import sys


sys.path.append(os.getcwd())

from bin import lib
import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)
    print(f'Running batch: {lib.get_batch_name()}, with output folder: {lib.get_batch_output_folder()}')
    logging.info(f'Running batch: {lib.get_batch_name()}, with output folder: {lib.get_batch_output_folder()}')


    img = cv2.imread('messi5.jpg')
    print(img)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()



# Main section
if __name__ == '__main__':
    main()
