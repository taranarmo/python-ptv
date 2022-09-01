import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import pims
import cv2
import time
import os
from threading import Thread
from queue import Queue, Empty

VIDEO_FILE = ""
TEMP_DIRECTORY = "/tmp/opencv_subtract_bg"
THREAD_COUNT = 8

if not os.path.exists(TEMP_DIRECTORY):
    os.mkdir(TEMP_DIRECTORY)

def subtract_backgroung(video_file, subtractor=cv2.createBackgroundSubtractorMOG2):
    cap = cv2.VideoCapture(video_file)
    fgbg = subtractor()
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        filename = os.path.join(TEMP_DIRECTORY, f"frame{i}.png")
        cv2.imwrite(filename, fgmask)
        i += 1
    cap.release()


# Due to background and foreground separation grayscale is not needed
# @pims.pipeline
# def gray(image):
#     return image[:, :, 1]  # Take just the green channel
def track(frames=TEMP_DIRECTORY):
    if os.path.isdir(frames):
        frames = os.path.join(frames, "*.png")
    frames = pims.open(frames)
    queue = Queue()

    def populate_queue(frames, queue):
        for frame in frames:
            while queue.qsize() > 30:
                time.sleep(1)
            else:
                queue.put(frame)
        print("frames are done")

    producer_thread = Thread(
            target=populate_queue,
            args=(frames, queue)
            )
    producer_thread.start()

    def process_queue(queue):
        while True:
            try:
                frame = queue.get(timeout=5)
                print(f"processing frame {frame.frame_no}")
                tp.locate(frame, 55, minmass=20)
            except Empty:
                break

    workers = []
    for i in range(THREAD_COUNT):
        worker = Thread(
                target=process_queue,
                args=(queue,),
                )
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    queue.join()

def main():
    subtract_backgroung(VIDEO_FILE)
    track(TEMP_DIRECTORY)

if __name__ == "__main__":
    main()
# tp.motion.compute_drift(tp.link(f, 30))
