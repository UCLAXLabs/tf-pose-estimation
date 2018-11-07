import argparse
import logging
import time

import cv2
import numpy as np

import json

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
#from tf_pose import common

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
    parser.add_argument('--resolution', type=str, default='0x0', help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    cap = cv2.VideoCapture(args.video)

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
    while cap.isOpened():
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
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            print(len(humans),"figures detected")
            figures = TfPoseEstimator.get_figures(image, humans)
        except:
            print("Error with inference")
            break

        print(len(humans),"humans detected")
        if not args.showBG:
            image = np.zeros(image.shape)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #image = TfPoseEstimator.draw_human_clouds(image, humans, imgcopy=False)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.imshow('tf-pose-estimation result', image)
        cv2.imwrite('video_figures/' + str(fps_time) + '.jpg', image)
        fps_time = time.time()

        timeFigures[str(frameId)] = figures

        with open("figures.json", "a") as figuresFile:
            outStr = json.dumps(timeFigures)
            if (not firstFrame):
                figuresFile.write("," + outStr + "\n")
            else:
                figuresFile.write(outStr + "\n")
                firstFrame = False
        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

with open("figures.json", "a") as figuresFile:
    figuresFile.write("\n]")


logger.debug('finished+')
