import logging
import yacs.config
from gaze_estimation.gaze_estimator.common import (Face,
                                                   Visualizer)
from gaze_estimation.utils import load_config
from gaze_estimation import GazeEstimationMethod, GazeEstimator
import time
import draw_utils
from helper_fn import point_to_screen
from screen_conf import *
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from datetime import datetime

logging.basicConfig(filename=f'log/Gaze_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.log',
                            filemode='a',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG
                    )
logger = logging.getLogger(__name__)


# AVERAGING OVER LANDMARKS TOGGLE
AVG_LANDMARKS = 0
num_avg_frames = 3 # num of frames to average over


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
            print(objectCentroids, inputCentroids)
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


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):
        self.tracker = CentroidTracker()
        self.config = config
        self.gaze_estimator = GazeEstimator(config, AVG_LANDMARKS=AVG_LANDMARKS, num_frames=num_avg_frames)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        self.cap = self._create_capture()

        self.stop = False

        # FRAME COUNTER
        self.i = 0
        self.pts = []
        self.cur_pos = []
        self.true_pos = []
        self.face_gaze = []
        self.face_cent = []

    def run(self) -> None:
        while True:
            print((W_px, adj_H))
            img1 = cv2.imread("billboard.jpg")
            img = cv2.resize(img1, (W_px, adj_H), interpolation=cv2.INTER_AREA)

            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    print(self.tracker.viewers)
                    logger.info(str(self.tracker.viewers))
                    # print(self.pts)
                    # print(np.array([[point[0], point[1]] for point in self.pts]).shape)
                    logger.info(f"View count = {len(list(filter(lambda x: x > 50, self.tracker.viewers.values())))}")
                    break

            response, frame = self.cap.read()
            if not response:
                break
            # FIRST WE UNDISTORT THE IMAGE!
            undistorted = cv2.undistort(
                frame, self.gaze_estimator.camera.camera_matrix,
                self.gaze_estimator.camera.dist_coefficients)

            faces = self.gaze_estimator.detect_faces(undistorted)
            self.visualizer.set_image(frame.copy())
            recs = []
            points = []
            for face in faces:
                bbx = face.bbox.flatten()
                self.gaze_estimator.estimate_gaze(undistorted, face)
                mid_point = self._draw_gaze_vector(face)
                recs.append(bbx)
                pts = draw_utils.display_canv(CANV_MODE=CANV_MODE, cur_pos=mid_point)
                self.pts.append(pts)
                self.true_pos.append(pts[0])
                self.cur_pos.append(pts[1])
                cv2.circle(img, (mid_point[0], mid_point[1]), 4, (0, 255, 0), -1)
                print(f"x[{mid_point[0]}:{W_px}] \n y[{mid_point[1]}:{H_px}]")
                if 0 <= mid_point[0] <= W_px and 0 <= mid_point[1] <= adj_H:
                    points.append(True)
                else:
                    points.append(False)

            cv2.imshow('points', img)
            cv2.moveWindow('points', 0, 0)
            objects = self.tracker.update((recs, points))
            temp = np.zeros(frame.shape)

            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(temp, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            temp = temp[:, ::-1]
            temp = cv2.resize(temp, (0, 0), fy=IMG_SCALE, fx=IMG_SCALE)
            cv2.imshow("tracker", temp)
            cv2.moveWindow("tracker", 0, temp.shape[0])
            if self.config.demo.use_camera:
                self.visualizer.image = self.visualizer.image[:, ::-1]

            if self.config.demo.display_on_screen:
                self.visualizer.image = cv2.resize(self.visualizer.image, (0, 0), fy=IMG_SCALE, fx=IMG_SCALE)
                cv2.imshow('frame', self.visualizer.image)
                # MOVE TO TOP LEFT CORNER
                cv2.moveWindow("frame", 0, 0)
            self.i += 1
        self.cap.release()

    def _wait_key(self) -> None:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True

    def _create_capture(self) -> cv2.VideoCapture:
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        # pdb.set_trace()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        # print(self.gaze_estimator.camera.width, self.gaze_estimator.camera.height)
        return cap

    def _draw_gaze_vector(self, face: Face):
        length = self.config.demo.gaze_visualization_length
        print('*' * 50)
        if self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            self.face_cent.append(face.center)
            self.face_gaze.append(face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError

        if self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            XY = point_to_screen(face.center, face.gaze_vector)
            mid_x = XY[0]
            mid_y = XY[1]

        else:
            raise ValueError

        mid_point = (int(mid_x), int(mid_y))
        logger.info(f'[point] x: {mid_point[0]}, y: {mid_point[1]}')
        return mid_point


def main():
    global CANV_MODE, IMG_SCALE
    start_time = time.time()
    config, custom = load_config('configs/demo_mpiifacegaze_resnet_simple_14.yaml')
    IMG_SCALE = custom['imgscale']
    CANV_MODE = custom['mode']  # 'RNG'
    demo = Demo(config)
    demo.run()
    n_frames = len(demo.pts)
    tot_time = time.time() - start_time
    logger.info(f'nr of frames: {n_frames}')
    logger.info(f'All finished: {tot_time} seconds.')
    logger.info(f'FPS: {round(n_frames / tot_time, 2)}')


if __name__ == '__main__':
    main()
