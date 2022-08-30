import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import pims
import cv2


cap = cv2.VideoCapture(video_file)
fgbg = cv2.createBackgroundSubtractorMOG2()

for i in range(100):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imwrite(f"Downloads/masks/frame{i}.png", fgmask)
    k = cv2.waitKey(30) & 0xff
    if k==ord('q'):
        break


# Due to background and foreground separation grayscale is not needed
# @pims.pipeline
# def gray(image):
#     return image[:, :, 1]  # Take just the green channel
frames = gray(pims.open('Downloads/masks/out.mp4'))

tp.motion.compute_drift(tp.link(f, 30))
