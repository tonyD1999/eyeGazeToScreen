# BILLBOARD ATTRACTION
> We will use camera to count how many people looked your billboard to answer the
question that “Is your billboard attractive ?”
> Using Dlib and MPIIFaceGaze

## Calibration values
* Setting height, width, camera
> check **screen_conf.py** for more details

## Model weights
* Landmarks: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
* Face model: https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiifacegaze_resnet_simple.pth
* My Face model: https://drive.google.com/drive/folders/10AIdFyfSh2lj4iqnDpJCmL18F0TndFIR?usp=sharing
    * model.pth: my model
The Landmarks dlib file needs to be unzipped and moved to **pytorch_mpiigaze/data/dlib/** folder. Move the pth files into **mpiifacegaze/resnet_simple_14/** for the face model.

## Demo
* Gaze Estimation Demo
    ```
        python mydemo.py
    ```
* Streamlit + Flask
    ```
        set FLASK_APP=flask_server.py
        flask run --host=<your ip> --port:<your defined port>
        streamlit run streamlit_demo.py
    ```
    * Note:
        * You need to change url in streamlit_demo.py
        ```
            url = "http://<server_ip>:<server_port>/"
        ```

## Result:
* Gaze Estimation Demo
    * Camera frame:
        * ![camera_frame](https://i.imgur.com/SBdDF2H.png)
    * Face tracker:
        ![tracker](https://i.imgur.com/icJDIxw.png)
    * Points on screen:
        * ![Points](https://drive.google.com/uc?export=view&id=1K_nHApgI5bZ_F0hyXrDIhYPNRuzTqEMT)
    * Grid View:
        * ![Grid](https://i.imgur.com/PZ7PrjI.jpg)

* Streamlit + Flask Demo:
    * Run Demo:
        * ![Streamlit Demo](https://i.imgur.com/2jKXwfX.png)

    * Log View:
        * ![Log](https://i.imgur.com/cXBnWMX.png)

    * Point View:
        * ![Point](https://i.imgur.com/lMAogNJ.png)

    * Viewer:
        * ![Viewer](https://i.imgur.com/OAEKiTb.png)


## Requirements

* Python >= 3.7

```bash
pip install -r requirements.txt
```


## Download the dataset and preprocess it
### MPIIFaceGaze
```bash
bash scripts/download_mpiifacegaze_dataset.sh
python tools/preprocess_mpiifacegaze.py --dataset datasets/MPIIFaceGaze -o datasets/
```
* **Note**:
    * Colab has limited storage so that dataset consumes lots of space. To overcome this problem, you should make a shortcut of this dataset (Google Drive)
        * https://drive.google.com/drive/folders/19Z1Z1hELyeJmgrm5RcgIqv3ax2HQ7_T3?usp=sharing
    * Then preprocess group of files

## Usage

* This repository uses [YACS](https://github.com/rbgirshick/yacs) for
configuration management.
* Default parameters are specified in `gaze_estimation/config/defaults.py` (which is not supposed to be modified directly).
* You can overwrite those default parameters using a YAML file like `configs/mpiigaze/alexnet_train.yaml`.


### Training and Evaluation
* Train a model using all the data except the person with ID 0, and run test on that person.
    * Looking **MPIIFaceGaze.ipynb** for infomation


## Results
### MPIIFaceGaze

| Model     | Mean Test Angle Error [degree] | Training Time |
|:----------|:------------------------------:|--------------:|
| AlexNet   |              5.06              |  135 s/epoch  |
| ResNet-14 |              4.83              |   62 s/epoch  |

The training time is the value when using GTX 1080Ti.

![AlexNet](https://i.imgur.com/HZ6t3Jm.png)


![ResNet-14](https://i.imgur.com/obftO7O.png)


### My MPIIFaceGaze

| Model     |     Test Angle Error [degree]  | Training Time |
|:----------|:------------------------------:|--------------:|
| AlexNet   |              2.63              |  780 s/epoch  |
| ResNet-14 |              2.17              |  780 s/epoch  |

The training time is the value when using Colab


## References

* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
* https://github.com/kenkyusha/eyeGazeToScreen


