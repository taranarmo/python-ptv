import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import pims
import cv2
import time
import os
import argparse
import pathlib
from threading import Thread
from queue import Queue, Empty

parser = argparse.ArgumentParser('Track particles on video')
parser.add_argument('video_file')
parser.add_argument(
        '--denoise',
        help='apply denoise filter to the frames',
        action='store_true'
        )
parser.add_argument(
        '--threads',
        type=int,
        help='threads to use',
        default=os.cpu_count()
        )
parser.add_argument(
        '--temp_directory',
        type=pathlib.Path,
        help='directory to store temporary frames',
        default='/tmp/opencv_subtract_bg'
        )
parser.add_argument(
        '--save_directory',
        type=pathlib.Path,
        help='directory to save result files',
        default=os.curdir
        )
parser.add_argument(
        '--denoise_parameter',
        type=int,
        help='parameter h passed to fastNlMeansDenoisingColored of openCV',
        default=2,
        dest='denoise_parameter',
        )
args = parser.parse_args()

VIDEO_FILE = args.video_file
TEMP_DIRECTORY = args.temp_directory
SAVE_DIRECTORY = args.save_directory
THREAD_COUNT = args.threads
DENOISE = args.denoise
DENOISE_PARAMETER = args.denoise_parameter

for directory in (TEMP_DIRECTORY, SAVE_DIRECTORY):
    if not os.path.exists(directory):
        os.mkdir(directory)

def subtract_backgroung(
        video_file,
        subtractor=cv2.createBackgroundSubtractorMOG2,
        denoise=DENOISE,
        h=DENOISE_PARAMETER,
        ):
    cap = cv2.VideoCapture(video_file)
    fgbg = subtractor()
    if denoise:
        process_image = lambda x: fgbg.apply(
                cv2.fastNlMeansDenoisingColored(x, h=h)
                )
    else:
        process_image = lambda x: fgbg.apply(x)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = process_image(frame)
        filename = os.path.join(TEMP_DIRECTORY, f'frame{i}.png')
        cv2.imwrite(filename, fgmask)
        i += 1
    cap.release()


# Due to background and foreground separation grayscale is not needed
# @pims.pipeline
# def gray(image):
#     return image[:, :, 1]  # Take just the green channel
def track(frames=TEMP_DIRECTORY):
    if os.path.isdir(frames):
        frames = os.path.join(frames, '*.png')
    frames = pims.open(frames)
    queue = Queue()

    def populate_queue(frames, queue):
        for frame in frames:
            while queue.qsize() > 30:
                time.sleep(1)
            else:
                queue.put(frame)
        print('frames are done')

    producer_thread = Thread(
            target=populate_queue,
            args=(frames, queue)
            )
    producer_thread.start()

    def process_queue(queue):
        while True:
            try:
                frame = queue.get(timeout=5)
                print(f'processing frame {frame.frame_no}')
                particles = tp.locate(frame, 55, minmass=20)
                particles.reset_index().to_feather(
                        os.path.join(
                            SAVE_DIRECTORY,
                            f'particles_frame_{frame.frame_no}.feather'
                            )
                        )
                queue.task_done()
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

if __name__ == '__main__':
    main()
# tp.motion.compute_drift(tp.link(f, 30))
