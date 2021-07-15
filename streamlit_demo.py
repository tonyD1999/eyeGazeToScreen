import streamlit as st
import cv2
import pickle
import requests
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    st.title('Gaze Estimation')
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    display = st.empty()
    url = "http://10.11.133.153:5001/"
    tick = st.checkbox("ON/OFF")
    if tick:
        screen_size = requests.post(url+'start', data=pickle.dumps({'message': f'Billboard_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")} STARTED'}, protocol=pickle.HIGHEST_PROTOCOL)).json()['screen_size']
        print(screen_size)
        while True:
            ret, frame = cam.read()
            data = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
            resp_data = requests.post(url+'detect', data=data).json()
            img1 = cv2.imread("billboard.jpg")
            img = cv2.resize(img1, tuple(screen_size), interpolation=cv2.INTER_AREA)
            points = resp_data['points']
            for point in points:
                cv2.circle(img, (point[0], point[1]), 4, (0, 255, 0), -1)
            display.image(img, channels="BGR", use_column_width=True)
    else:
        requests.post(url + 'stop', data=pickle.dumps(
            {'message': f'Billboard_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")} STOPPED'},
            protocol=pickle.HIGHEST_PROTOCOL))
        LOG_ANALYSIS = st.checkbox('LOG')
        POINT_ANALYSIS = st.checkbox('POINT')
        VIEWER_ANALYSIS = st.checkbox('VIEWER')
        if LOG_ANALYSIS:
            get_info_data = requests.post(url + 'getinfo').json()
            log_select = st.selectbox("Log", [get_info_data['log_file']])
            print(log_select)
            if log_select:
                log = pickle.loads(requests.post(url + 'getfile', data=pickle.dumps(
                    {
                        'filename': log_select
                    },
                    protocol=pickle.HIGHEST_PROTOCOL
                )).content)
                st.text_area('','\n'.join(log), 500)

        if POINT_ANALYSIS:
            get_info_data = requests.post(url + 'getinfo').json()
            point_file = st.selectbox("Points", [get_info_data['point_file']])
            print(point_file)
            if point_file:
                data = pickle.loads(requests.post(url + 'getfile', data=pickle.dumps(
                    {
                        'filename': point_file
                    },
                    protocol=pickle.HIGHEST_PROTOCOL
                )).content)
                points = data['data']
                screen_size = data['screen_size']
                x = [point[0] for point in points]
                y = [-point[1] for point in points]
                heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                plt.clf()
                plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet)
                plt.colorbar()
                st.pyplot()
        if VIEWER_ANALYSIS:
            get_info_data = requests.post(url + 'getinfo').json()
            viewer_file = st.selectbox("Viewer", [get_info_data['viewer_file']])
            if viewer_file:
                data = pickle.loads(requests.post(url + 'getfile', data=pickle.dumps(
                    {
                        'filename': viewer_file
                    },
                    protocol=pickle.HIGHEST_PROTOCOL
                )).content)['data']
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])
                user_ids = [f'User {key}' for key in data.keys()]
                count = list(data.values())
                ax.bar(user_ids, count)
                ax.set_ylabel('Frame count')
                st.pyplot()





main()