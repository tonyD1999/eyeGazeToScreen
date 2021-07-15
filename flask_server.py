import logging
from gaze_estimation.gaze_estimator.common import Face
from gaze_estimation.utils import load_config
from gaze_estimation import GazeEstimationMethod, GazeEstimator
import time
from helper_fn import point_to_screen
from screen_conf import *
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from datetime import datetime
from flask import Flask, request
import json
import pickle
from os import listdir
from os.path import isfile, join

logging.basicConfig(filename=f'log/Gaze_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.log',
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG
                        )
logger = logging.getLogger(__name__)

# AVERAGING OVER LANDMARKS TOGGLE
AVG_LANDMARKS = 0
num_avg_frames = 3  # num of frames to average over
flag = False


class CentroidTracker:
    def __init__(self, maxDisappeared=100):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.viewers = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, view):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        if view:
            self.viewers[self.nextObjectID] = 1
        else:
            self.viewers[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def reset(self):
        self.objects.clear()
        self.disappeared.clear()
        self.viewers.clear()
        self.nextObjectID = 0

    def update(self, rects):
        if len(rects[0]) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects[0]), 2), dtype="int")
        points = rects[1]
        for (i, (startX, startY, endX, endY)) in enumerate(rects[0]):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], points[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # print(objectCentroids, inputCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                if points[col]:
                    self.viewers[objectID] += 1
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], points[col])
        return self.objects


config, custom = load_config('configs/demo_mpiifacegaze_resnet_simple_14.yaml')
IMG_SCALE = custom['imgscale']
CANV_MODE = custom['mode']  # 'RNG'

tracker = CentroidTracker()
gaze_estimator = GazeEstimator(config, AVG_LANDMARKS=AVG_LANDMARKS, num_frames=num_avg_frames)

# FRAME COUNTER
i = 0

point_storage = []

app = Flask(__name__)

start_time = 0


@app.route('/start', methods=['POST'])
def start():
    if request.method == 'POST':
        global start_time, i, point_storage, flag
        i = 0
        start_time = time.time()
        point_storage.clear()
        tracker.reset()
        data = pickle.loads(request.data)
        logger.info(data['message'])

        flag = True
    return json.dumps({'screen_size': (W_px, adj_H)})


@app.route('/detect', methods=['POST'])
def detect():
    global i
    if request.method == 'POST':
        frame = pickle.loads(request.data)
        undistorted = cv2.undistort(
            frame, gaze_estimator.camera.camera_matrix,
            gaze_estimator.camera.dist_coefficients)

        faces = gaze_estimator.detect_faces(undistorted)
        recs = []
        points = []
        pts = []
        for face in faces:
            bbx = face.bbox.flatten()
            gaze_estimator.estimate_gaze(undistorted, face)
            mid_point = _draw_gaze_vector(face)
            point_storage.append(mid_point)
            recs.append(bbx)
            pts.append(mid_point)
            # print(f"x[{mid_point[0]}:{W_px}] \n y[{mid_point[1]}:{H_px}]")
            if 0 <= mid_point[0] <= W_px and 0 <= mid_point[1] <= adj_H:
                points.append(True)
            else:
                points.append(False)

        tracker.update((recs, points))
        i += 1
        return json.dumps({'points': pts})


@app.route('/stop', methods=['POST'])
def stop():
    if request.method == 'POST':
        global start_time, i, point_storage, flag
        if flag:
            flag = False
            tot_time = time.time() - start_time
            get_data = pickle.loads(request.data)
            logger.info(get_data["message"])
            view_count = len(list(filter(lambda x: x > 50, tracker.viewers.values())))
            logger.info(f"View count = {view_count}")
            logger.info(f'nr of frames: {i}')
            logger.info(f'All finished: {tot_time} seconds.')
            logger.info(f'FPS: {round(i / tot_time, 2)}')

            with open(f'viewer_info/Viewers_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pickle', 'wb') as handle:
                pickle.dump(tracker.viewers, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'viewer_info/Points_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pickle', 'wb') as handle:
                pickle.dump(point_storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
            i = 0
            tracker.reset()
            point_storage.clear()
    return "Ok"


@app.route('/getinfo', methods=['POST'])
def get_info():
    if request.method == 'POST':
        log_file = [f for f in listdir('log') if isfile(join("log", f))]
        pickle_file = [f for f in listdir('viewer_info') if isfile(join('viewer_info', f))]
        point_file = list(filter(lambda x: "Points" in x, pickle_file))
        viewer_file = list(filter(lambda x: "Viewers" in x, pickle_file))
        data = {'log_file': log_file[-1],
                'point_file': point_file[-1],
                'viewer_file': viewer_file[-1]
                }
        return json.dumps(data)


@app.route('/getfile', methods=['POST'])
def get_file():
    if request.method == 'POST':
        get_data = pickle.loads(request.data)
        file_name = get_data['filename']
        print(file_name)
        if 'pickle' in file_name:
            with open(join('viewer_info', file_name), 'rb') as handle:
                data = {'data': pickle.load(handle), 'screen_size': (W_px, adj_H)}
        elif 'log' in file_name:
            file = open(join('log', file_name))
            data = file.readlines()

        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def _draw_gaze_vector(face: Face):
    if config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
        logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        XY = point_to_screen(face.center, face.gaze_vector)
        mid_x = XY[0]
        mid_y = XY[1]

    else:
        raise ValueError

    mid_point = (int(mid_x), int(mid_y))
    logger.info(f'[point] x: {mid_point[0]}, y: {mid_point[1]}')
    return mid_point


# app.run(host='192.168.31.184', port='5001', debug=True)







