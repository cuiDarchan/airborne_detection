######################################################################################
### This is a read-only file to allow participants to run their code locally.      ###
### It will be over-writter during the evaluation, Please do not make any changes  ###
### to this file.                                                                  ###
######################################################################################

import json
import traceback
import os
import signal
from contextlib import contextmanager
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import cm
from utility.vis_writer import VisWriter
from evaluator import aicrowd_helpers
import cv2
from tqdm import tqdm

class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Prediction timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class AirbornePredictor:
    def __init__(self):
        self.test_data_path = os.getenv("TEST_DATASET_PATH", os.getcwd() + "/data/part1/Images/")
        self.inference_output_path = self.get_results_directory()
        self.inference_setup_timeout = int(os.getenv("INFERENCE_SETUP_TIMEOUT_SECONDS", "600"))
        self.inference_flight_timeout = int(os.getenv("INFERENCE_PER_FLIGHT_TIMEOUT_SECONDS", "600"))
        self.partial_run = os.getenv("PARTIAL_RUN_FLIGHTS", None)
        self.results = []
        self.current_flight_results = []
        self.current_img_name = None
        self.track_id_seq = 0
        self.frame_result_list = []
        self.dump_result_video = True
        # by default, 50 colors
        self.num_colors = 50
        self.colors = self.get_n_colors(self.num_colors)
        self.output_path = os.getenv("TEST_DATASET_PATH", os.getcwd() + "/data/results/")
        self.vis_writer = VisWriter(dump_video=self.dump_result_video,
                           out_path=self.output_path,
                           file_name=os.path.basename('test.mp4'))
        self.ori_vis_writer = VisWriter(dump_video=self.dump_result_video,
                           out_path=self.output_path,
                           file_name=os.path.basename('origin.mp4'))
        self.last_result_size = 0
        
    def get_frame_result_list(self):
        return self.frame_result_list
        
    def register_object_and_location(self, class_name, track_id, bbox, confidence, img_name):
        """
        Register all your tracking results to this function.
           `track_id`: unique id for your detected airborne object
           `img_name`: The image_name to which this prediction belongs.
        """
        assert 0 < confidence < 1
        assert track_id is not None
        if img_name is None:
            img_name = self.current_img_name

        result = {
            "detections": [
                {
                    "track_id": track_id,
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2],
                    "h": bbox[3],
                    "n": class_name,
                    "s": confidence
                }
            ],
            "img_name": img_name
        }
        self.results.append(result)
        self.current_flight_results.append(result)

    ## 获取data目录下，所有images内的航线flight_ids
    def get_all_flight_ids(self):
        valid_flight_ids = None
        if self.partial_run:
            valid_flight_ids = self.partial_run.split(',')
        flight_ids = []
        for folder in listdir(self.test_data_path):
            if not isfile(join(self.test_data_path, folder)):
                if valid_flight_ids is None or folder in valid_flight_ids:
                    flight_ids.append(folder)
        return flight_ids

    def get_all_frame_images(self, flight_id):
        frames = []
        flight_folder = join(self.test_data_path, flight_id)
        for frame in listdir(flight_folder):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)
        return frames

    ## 将数据地址、flight_id、frame_id组成一个图片的地址
    def get_frame_image_location(self, flight_id, frame_id):
        return join(self.test_data_path, flight_id, frame_id)

    def get_flight_folder_location(self, flight_id):
        return join(self.test_data_path, flight_id)

    def evaluation(self):
        """
        Admin function: Runs the whole evaluation
        """
        aicrowd_helpers.execution_start()
        try:
            with time_limit(self.inference_setup_timeout):
                self.inference_setup()
        except NotImplementedError:
            print("inference_setup doesn't exist for this run, skipping...")

        aicrowd_helpers.execution_running()

        flights = self.get_all_flight_ids()

        for flight in flights:
            with time_limit(self.inference_flight_timeout):
                self.inference(flight)
                
            self.save_results(flight)

        self.save_results()
        aicrowd_helpers.execution_success()

    def run(self):
        try:
            self.evaluation()
        except Exception as e:
            error = traceback.format_exc()
            print(error)
            aicrowd_helpers.execution_error(error)
            if not aicrowd_helpers.is_grading():
                raise e

    def inference_setup(self):
        """
        You can do any preprocessing required for your codebase here : 
            like loading your models into memory, etc.
        """
        raise NotImplementedError

    def inference(self, flight_id):
        """
        This function will be called for all the flight during the evaluation.
        NOTE: In case you want to load your model, please do so in `inference_setup` function.
        """
        raise NotImplementedError

    def get_results_directory(self, flight_id=None):
        """
        Utility function: results directory path
        """
        root_directory = os.getenv("INFERENCE_OUTPUT_PATH", os.getcwd() + "/data/results/")
        run_id = os.getenv("DATASET_ENV", "run0")
        results_directory = os.path.join(root_directory, run_id)
        if flight_id is not None:
            results_directory = os.path.join(results_directory, flight_id)
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        return results_directory

    def save_results(self, flight_id=None):
        """
        Utility function: save results in nested direcotry based on flight_id
        This helps in giving continuous feedback based on of the approximate 
        scores.
        """
        if flight_id is None:
            submission = self.results
        else:
            submission = self.current_flight_results

        with open(os.path.join(self.get_results_directory(flight_id), "result.json"), 'w') as fp:
            json.dump(submission, fp)

        self.current_flight_results = []
    

    @staticmethod
    def get_n_colors(n, colormap="gist_ncar"):
        # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
        # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
        # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
        # For more options see:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # and https://matplotlib.org/users/colormaps.html

        colors = cm.get_cmap(colormap)(np.linspace(0, 1, n))
        # Randomly shuffle the colors
        np.random.shuffle(colors)
        # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
        # Also multiply by 255 since cm returns values in the range [0, 1]
        colors = colors[:, (2, 1, 0)] * 255
        return colors
    
    
    ## 将标注框保存到图像上
    def frame_vis_generator(self, frame, results, mode='RGB'):        
        if mode =='GRAY': 
            frame =cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        size = len(results) # results 结果在断断续续+1
        if size > 0 :
            result = results[0] # 取最新的
            track_id = int(result['track_id'])
            cx = result['cx'] + result['offset'][0]
            cy = result['cy'] + result['offset'][1]
            w = float(result['w'])
            h = float(result['h'])
            class_name = 'airborne'
            confidence = float(result['conf'])
            distance = float(result['distance'])
            x1, y1, x2, y2 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)
            
            color = (0, 0, 255) # 红色
            # color = self.colors[track_id % self.num_colors] # 随机色
            text_width = len(class_name)*16
            confidence_width = 80
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)
            cv2.putText(frame, str(track_id), (x1 + 5, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)
            # 矩形在框的上部分，内部写种类
            cv2.rectangle(frame, (x1, y1-30), (x1+text_width, y1), color, -1)
            # cv2.putText(frame, str(class_name), (x1 + 5, y1 - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            cv2.putText(frame, '{:.2f}'.format(distance), (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            # 矩形在框的下部分，内部写种类
            cv2.rectangle(frame, (x1, y2+1), (x1+confidence_width, y2+30), color, -1)
            cv2.putText(frame, '{:.2f}'.format(confidence), (x1 + 5, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            self.last_result_size = size
            
            ## debug save test img
            #cv2.imwrite('/home/cui/workspace/airborne-detection-starter-kit/data/results/test2.jpg', frame)
        return frame
    
    ## 保存结果到mp4格式数据中，mode 支持'GRAY'和'RGB'
    def save_result_video(self, mode = 'GRAY'):
        if self.dump_result_video:
            self.vis_writer._init_video_writer(mode = mode)
            results = self.get_frame_result_list()
            for result in results:
                self.vis_writer._video_writer.write(result)
            self.vis_writer.close_video_writer()
    
    ## 保存原视频,按RGB方式
    def save_origin_video(self, mode='RGB'):
        self.ori_vis_writer._init_video_writer(mode = mode)
        flights = self.get_all_flight_ids()
        for flight_id in flights:
            for frame_image in tqdm(self.get_all_frame_images(flight_id)):
                frame_image_path = self.get_frame_image_location(flight_id, frame_image)
                frame = cv2.imread(frame_image_path)
                self.ori_vis_writer._video_writer.write(frame)
        self.ori_vis_writer.close_video_writer()
                
    def get_results(self):
        return self.results