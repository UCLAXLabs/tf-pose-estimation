import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import json

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    #logger.debug('cam read+')
    #cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(args.video)
    #ret_val, image = cap.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    targetFramerate = 30 # fps

    fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second

    skipRatio = int(round(float(fps) / float(targetFramerate)))
    
    outputFrameDuration = 1 / float(targetFramerate) # .0333333 ...
    sourceFrameDuration = 1 / float(fps) # .016672224 ...

    #skipRemainder = float(targetFramerate) - (float(fps) % float(targetFramerate))

    with open("figures.json", "w") as figuresFile:
        figuresFile.write("[\n")
    figuresFile.close()

    firstFrame = True

    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    outputTimecode = 0
    sourceTimecode = 0

    frameCount = 0
    while(cap.isOpened()):

        ret_val, image = cap.read()

        frameCount += 1

        sourceTimecode += sourceFrameDuration

        frameId = int(round(cap.get(1))) 

        if ((frameCount % skipRatio) != 0):

            if ((sourceTimecode - outputTimecode) > outputFrameDuration):
                print("sourceTimecode",sourceTimecode,"outputTimecode",outputTimecode,"outputFrameDuration",outputFrameDuration,"not skipping")
            else:
                print("skipping frame",frameId,"count",frameCount,"skip rato",skipRatio)
                continue

        print("processing frame",frameId,"count",frameCount)

        outputTimecode += outputFrameDuration

        fps_time = time.time()

        timeFigures = {}

        try:
            humans = e.inference(image)
        except:
            break

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #image = TfPoseEstimator.draw_human_clouds(image, humans, imgcopy=False)

        figures = TfPoseEstimator.get_figures(image, humans)

        timeFigures[str(frameId)] = figures

        #logger.debug('show+')
        cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
        #cv2.imshow('tf-pose-estimation result', image)
        cv2.imwrite('video_figures/' + str(fps_time) + '.jpg', image)
        fps_time = time.time()
        #if cv2.waitKey(1) == 27:
        #    break
        with open("figures.json", "a") as figuresFile:
            outStr = json.dumps(timeFigures)
            if (not firstFrame):
                figuresFile.write("," + outStr + "\n")
            else:
                figuresFile.write(outStr + "\n")
                firstFrame = False

    cv2.destroyAllWindows()

with open("figures.json", "a") as figuresFile:
    figuresFile.write("\n]")


logger.debug('finished+')
