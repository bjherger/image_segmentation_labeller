#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
Key 'j' - Last image
Key 'k' - Next image
Key 'q' - change brush to size 5 (default)
Key 'w' - change brush to size 10
Key 'e' - change brush to size 30
Key 'r' - change brush to size 60

===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import glob
import logging
import os

import numpy as np
import cv2 as cv

import sys

class App():
    BLUE = [255,0,0]        # rectangle color
    RED = [0,0,255]         # PR BG
    GREEN = [0,255,0]       # PR FG
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
    DRAW_PR_BG = {'color' : RED, 'val' : 2}

    def onmouse(self, event, x, y, flags, param):

        # draw touchup curves

        if event == cv.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

    def reset(self):
        print("resetting \n")

        logging.info(f'Reset current image index: {self.image_index}')

        filename = self.image_paths[self.image_index]
        self.img = cv.imread(cv.samples.findFile(filename))
        self.rect = tuple([0, 0] + list(self.img.shape[:2]))
        self.drawing = False
        self.rectangle = False
        self.rect_or_mask = 0
        self.rect_over = True
        self.value = self.DRAW_FG
        self.unaltered_image = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
        self.output = np.zeros(self.img.shape, np.uint8)  # output image to be shown

        # Perform initial segmentation
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv.grabCut(self.unaltered_image, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
        self.rect_or_mask = 1

    def save(self):

        output_base = os.path.splitext(os.path.basename(self.image_paths[self.image_index]))[0]
        output_full_file_path = f'data/output/{output_base}_mask.png'
        output_mask_file_path = f'data/output/{output_base}_mask.png'

        bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
        res = np.hstack((self.unaltered_image, bar, self.img, self.output))
        cv.imwrite(output_full_file_path, res)

        mask = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
        cv.imwrite(output_mask_file_path, mask)
        print(" Result saved as image \n")

    def run(self):
        logging.getLogger().setLevel(logging.INFO)

        # Loading images
        self.image_paths = glob.glob('data/input/images/*.png')
        logging.info(f'Image paths: {self.image_paths}')

        self.image_index = 0
        logging.info(f'Setting first image index: {self.image_index}')

        self.reset()

        # input and output windows
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.img.shape[1]+10,90)

        while(1):

            cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)

            # key bindings

            # ESC: Escape exts
            if k == 27:         # esc to exit
                break

            # 0: Background
            elif k == ord('0'): # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = self.DRAW_BG

            # 1: Foreground
            elif k == ord('1'): # FG drawing
                print(" mark foreground regions with left mouse button \n")
                self.value = self.DRAW_FG

            elif k == ord('q'):
                # Change brush to size 5 (default)
                self.thickness = 5

            elif k == ord('w'):
                # Change brush to size 10
                self.thickness = 10

            elif k == ord('e'):
                # Change brush to size 30
                self.thickness = 30

            elif k == ord('r'):
                # Change brush to size 60
                self.thickness = 60


            # Save image
            elif k == ord('s'): # save image
                self.save()

            elif k == ord('r'): # reset everything
                print('Saving before resetting')
                self.save()

                print("resetting \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                self.img = self.unaltered_image.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
                self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown

            # Perform one image segmentation setp
            elif k == ord('n'): # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                try:
                    if (self.rect_or_mask == 0):         # grabcut with rect
                        bgdmodel = np.zeros((1, 65), np.float64)
                        fgdmodel = np.zeros((1, 65), np.float64)
                        cv.grabCut(self.unaltered_image, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif self.rect_or_mask == 1:         # grabcut with mask
                        bgdmodel = np.zeros((1, 65), np.float64)
                        fgdmodel = np.zeros((1, 65), np.float64)
                        cv.grabCut(self.unaltered_image, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()

            # Reset
            elif k == ord('j'):
                print(f'Attempting to pull up last image index')
                if self.image_index >=1:
                    print(f'saving image at index:{self.image_index}, and pulling up last image')
                    self.save()
                    self.image_index -=1
                    self.reset()
                else:
                    print('Unable to pull up last image. ')

            elif k == ord('k'):
                print(f'Attempting to pull up next image index')
                if self.image_index + 1 <= len(self.image_paths) - 1:
                    print(f'saving image at index:{self.image_index}, and pulling up next image')
                    self.save()
                    self.image_index += 1
                    self.reset()
                else:
                    print('Unable to pull up next image. ')
                pass

            # Update output image
            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.output = cv.bitwise_and(self.unaltered_image, self.unaltered_image, mask=mask2)

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()