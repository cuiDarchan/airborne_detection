# This file is the entrypoint for your submission.
# You can modify this file to include your code or directly call your functions/modules from here.
import random
import cv2
from evaluator.airborne_detection import AirbornePredictor

from tqdm import tqdm

import sys
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.patches
import time

current_path = os.getcwd()
sys.path.append(f'{current_path}/seg_tracker')

from seg_tracker.seg_tracker import SegTracker, SegDetector, SegTrackerFromOffset

MIN_TRACK_LEN = 30
MIN_SCORE = 0.985


class SegPredictor(AirbornePredictor):
    training_data_path = None
    test_data_path = '/data/part1/Images/part1032b126da4eb4c58aa459e671c11f56b' # part100bb96a5a68f4fa5bc5c5dc66ce314d2
    vocabulary_path = None

    """
    推理设置
    PARTICIPANT_TODO:
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def inference_setup(self):
        self.detector = SegDetector() # 检测器
        self.tracker = SegTrackerFromOffset(detector=self.detector) #追踪器
        self.visualize_pred = False

    ## 获取某一航线flight_id下，所有图片的文件名，并排序
    def get_all_frame_images(self, flight_id):
        frames = []
        flight_folder = join(self.test_data_path, flight_id)
        for frame in sorted(listdir(flight_folder)):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)
        return frames

    def flight_started(self):
        self.track_id_results = {}
        self.visited_frame = {}
        self.track_len_so_far = {}
        self.tracker = SegTrackerFromOffset(detector=self.detector)

    def proxy_register_object_and_location(self, class_name, track_id, bbox, confidence, img_name):
        # MIN_TRACK_LEN check
        if track_id not in self.track_len_so_far:
            self.track_len_so_far[track_id] = 0
        self.track_len_so_far[track_id] += 1
        if self.track_len_so_far[track_id] <= MIN_TRACK_LEN:
            return

        # MIN_SCORE check
        if confidence < MIN_SCORE:
            return

        if img_name not in self.visited_frame:
            self.visited_frame[img_name] = []
        if track_id in self.visited_frame[img_name]:
            raise Exception('two entities  within the same frame {} have the same track id'.format(img_name))
        self.visited_frame[img_name].append(track_id)

        self.register_object_and_location(class_name, track_id, bbox, confidence, img_name)

    """
    PARTICIPANT_TODO:
    During the evaluation all combinations for flight_id and flight_folder_path will be provided one by one.
    """
    def inference(self, flight_id):
        self.flight_started()

        prev_frame = None
        
        for frame_image in tqdm(self.get_all_frame_images(flight_id)):
            frame_image_path = self.get_frame_image_location(flight_id, frame_image)
            frame = cv2.imread(frame_image_path, cv2.IMREAD_GRAYSCALE) #加载灰度图

            if prev_frame is None:
                prev_frame = frame
                continue

            results = self.tracker.predict(image=frame, prev_image=prev_frame)
            prev_frame = frame

            class_name = 'airborne'

            if self.visualize_pred and len(results):
                plt.imshow(frame)
                ax = plt.gca()

            for res in results:
                track_id = int(res['track_id'])
                confidence = float(res['conf'])
                cx = res['cx'] + res['offset'][0]
                cy = res['cy'] + res['offset'][1]
                w = res['w']
                h = res['h']

                bbox = [float(cx - w/2), float(cy - h/2), float(cx + w/2), float(cy + h/2)]
                # bbox = [float(cy - h / 2), float(cx - w / 2), float(cy + h / 2), float(cx + w / 2)]
                self.register_object_and_location(class_name, track_id, bbox, confidence, frame_image)

                if self.visualize_pred:
                    rect = matplotlib.patches.Rectangle(bbox[:2], w, h, linewidth=1, edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    plt.text(bbox[0], bbox[1], f'{int(confidence * 100)}', c='yellow')

            if self.visualize_pred and len(results):
                plt.show()
            
            
            ## 保存每一帧标定框到原始图像上
            if self.dump_result_video:
                frame_result = self.frame_vis_generator(frame, results, mode='GRAY')
                self.frame_result_list.append(frame_result)

# Transfer generated results to metrics codebase bbox format
def convert_and_copy_generated_results_to_metrics_folder():
    import json
    flight_results = json.loads(open("data/results/run0/result.json").read())
    for i in range(len(flight_results)):
        for j in range(len(flight_results[i]['detections'])):
            x = flight_results[i]['detections'][j]['x']
            y = flight_results[i]['detections'][j]['y']
            w = flight_results[i]['detections'][j]['w'] - x
            h = flight_results[i]['detections'][j]['h'] - y

            flight_results[i]['detections'][j]['x'] = x + w/2
            flight_results[i]['detections'][j]['y'] = y + h/2
            flight_results[i]['detections'][j]['w'] = w
            flight_results[i]['detections'][j]['h'] = h

    with open("data/evaluation/result/result.json", 'w') as fp:
        json.dump(flight_results, fp)


if __name__ == "__main__":
    submission = SegPredictor()
    submission.run()
    submission.save_result_video(mode='RGB') #保存MP4
    
    #submission.save_origin_video(mode='RGB') 
    # convert_and_copy_generated_results_to_metrics_folder()

