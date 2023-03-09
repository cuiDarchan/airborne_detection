import os
import logging
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import urllib
import zipfile
import matplotlib.pyplot as plt

import cv2
from . import models_segmentation
from . import models_transformation
from . import offset_grid_to_transform
from . import seg_prediction_to_items
from . import predict_ensemble
import common_utils

from prediction_structure import *
import tracking


class SegDetector:
    def __init__(self):
        models_dir = 'data/models'

        self.model_transform = models_transformation.TrEstimatorTsm(
            cfg=dict(
                base_model_name="resnet34",
                weight_scale=3
            ), pretrained=False)
        self.model_transform.load_state_dict(torch.load(f'{models_dir}/030_tr_tsn_rn34_w3_crop_borders_255_0.pth')["model_state_dict"])
        self.model_transform = self.model_transform.cuda()
        self.model_transform.eval()

        # self.model_seg = models_segmentation.DLASegmentation(
        #     cfg=dict(
        #         base_model_name="dla32",
        #         input_frames=2,
        #         feature_location='',
        #         combine_outputs_dim=512,
        #         combine_outputs_kernel=1,
        #         output_binary_mask=True,
        #         output_above_horizon=True,
        #         pred_scale=8
        #     ),
        #     pretrained=False
        # )
        #
        # self.model_seg.load_state_dict(torch.load(f'{models_dir}/100_dla32_fpm_1100.pth')['model_state_dict'], strict=False)

        self.model_seg1 = models_segmentation.HRNetSegmentation(
            cfg=dict(
                base_model_name="hrnet_w32",
                input_frames=2,
                feature_location='',
                combine_outputs_dim=512,
                combine_outputs_kernel=1,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8
            ),
            pretrained=False
        )

        self.model_seg1.load_state_dict(torch.load(f'{models_dir}/120_hrnet32_all_2220.pth')['model_state_dict'], strict=False)
        self.model_seg1 = self.model_seg1.cuda()
        self.model_seg1.eval()

        self.model_seg2 = models_segmentation.EffDetSegmentation8x(
            cfg=dict(
                base_model_name="tf_efficientdet_d2",
                custom_backbone="gernet_m",
                input_frames=2,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8
            ),
            pretrained=False
        )

        self.model_seg2.load_state_dict(torch.load(f'{models_dir}/120_gernet_m_b2_all_2220.pth')['model_state_dict'], strict=False)
        self.model_seg2 = self.model_seg2.cuda()
        self.model_seg2.eval()

        self.model_dla = models_segmentation.DLA8xSeparateHeads(
            cfg=dict(
                base_model_name="dla60_res2next",
                input_frames=2,
                feature_location='',
                combine_outputs_dim=256,
                combine_outputs_kernel=1,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8
            ),
            pretrained=False
        )

        self.model_dla.load_state_dict(torch.load(f'{models_dir}/120_dla60_256_sgd_all_rerun_2220.pth')['model_state_dict'], strict=False)
        self.model_dla = self.model_seg2.cuda()
        self.model_dla.eval()

        # self.model_seg3 = models_segmentation.EffDetSegmentation8x(
        #     cfg=dict(
        #         base_model_name="tf_efficientdet_d5",
        #         input_frames=2,
        #         output_binary_mask=True,
        #         output_above_horizon=True,
        #         pred_scale=8
        #     ),
        #     pretrained=False
        # )
        #
        # self.model_seg3.load_state_dict(torch.load(f'{models_dir}/120_edet_b5_all_2220.pth')['model_state_dict'], strict=False)
        # self.model_seg3 = self.model_seg3.cuda()
        # self.model_seg3.eval()

        self.model_seg3 = models_segmentation.HRNetSegmentation(
            cfg=dict(
                base_model_name="hrnet_w48",
                input_frames=2,
                feature_location='',
                combine_outputs_dim=512,
                combine_outputs_kernel=1,
                output_binary_mask=True,
                output_above_horizon=True,
                pred_scale=8
            ),
            pretrained=False
        )

        self.model_seg3.load_state_dict(torch.load(f'{models_dir}/130_hrnet48_all_2220.pth')['model_state_dict'], strict=False)
        self.model_seg3 = self.model_seg3.cuda()
        self.model_seg3.eval()

        self.frames = 0


    def estimate_transformation(self, prev_img, img):
        crop_w = 1024
        crop_h = 1024

        h, w = img.shape
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        img_crop = img[y0:y0 + crop_h, x0: x0 + crop_w]
        img_crop_prev = prev_img[y0:y0 + crop_h, x0: x0 + crop_w]

        img_crop_t = torch.from_numpy(img_crop / 255.0).float().cuda()
        img_crop_prev_t = torch.from_numpy(img_crop_prev / 255.0).float().cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                heatmap, offsets = self.model_transform(img_crop_prev_t[None, :], img_crop_t[None, :])
            heatmap = heatmap.detach().float().cpu().numpy()
            offsets = offsets.detach().float().cpu().numpy()

        prev_points = np.zeros((2, 32, 32), dtype=np.float32)
        prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
        prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]
        prev_points = prev_points[..., 2:-2, 2:-2]

        center_offset = np.array([512.0, 512.0])[:, None]

        cur_points = prev_points + offsets[0]
        dx, dy, angle, err = offset_grid_to_transform.offset_grid_to_transform_params(
            prev_frame_points=prev_points.reshape(2, -1) - center_offset,
            cur_frame_points=cur_points.reshape(2, -1) - center_offset,
            points_weight=heatmap[0].reshape(-1) ** 2
        )

        return dx, dy, angle

    '''
    估计整幅图的变换关系
    '''
    def estimate_transformation_full(self, prev_img, img):
        crop_w = 1024 + 1024
        crop_h = 1024 + 256

        crop_w_points = crop_w // 32  #64
        crop_h_points = crop_h // 32  #40

        h, w = img.shape #2048,2448
        y0 = 512  # (h - crop_h) // 2
        x0 = (w - crop_w) // 2  # 200
        img_crop = img[y0:y0 + crop_h, x0: x0 + crop_w] #(1280, 2048)
        img_crop_prev = prev_img[y0:y0 + crop_h, x0: x0 + crop_w] # 裁剪出中间部分

        img_crop_t = torch.from_numpy(img_crop / 255.0).float().cuda() # 归一化处理
        img_crop_prev_t = torch.from_numpy(img_crop_prev / 255.0).float().cuda()

        with torch.no_grad():
            '''
            torch.cuda.amp.autocast()作用：
            １）自动混合精度,自动将torch.FloatTensor类型转化为torch.HalfTensor
            ２）包含网络的前向过程(包括loss的计算),不要包含反向传播。
            '''
            with torch.cuda.amp.autocast(): 
                ## 推理两帧间的图像运动
                heatmap, offsets = self.model_transform(img_crop_prev_t[None, :], img_crop_t[None, :])
            heatmap = heatmap.detach().float().cpu().numpy() # 1,1,36,60
            offsets = offsets.detach().float().cpu().numpy() # 1,2,36,60

        # prev_points 两个维度x,y，类似图像上采样格网 40*64
        prev_points = np.zeros((2, crop_h_points, crop_w_points), dtype=np.float32) #2，40,64
        prev_points[0, :, :] = np.arange(16, crop_w, 32)[None, :] # [None, :]作用新增一个维度，从(64,)->(1,64)
        prev_points[1, :, :] = np.arange(16, crop_h, 32)[:, None]
        prev_points = prev_points[..., 2:-2, 2:-2] # padding

        points_offset = np.array([x0, y0])[:, None] # 2,1， x0=200, y0=512

        cur_points = prev_points + offsets[0] # 得到预测后的当前点
        ## 根据偏移后的中心部分格网offset_grid 计算 tr, 即两幅图对应采样点的线性变换关系
        tr, err = offset_grid_to_transform.offset_grid_to_transform(
            prev_frame_points=prev_points.reshape(2, -1) + points_offset,
            cur_frame_points=cur_points.reshape(2, -1) + points_offset,
            points_weight=heatmap[0].reshape(-1) ** 2
        )

        return tr
    
    '''
    目标检测主函数，参数为当前帧与上一帧图像
    '''
    def detect_objects(self, image, prev_image):
        batch_size = 1
        h, w = image.shape
        padding = (16 + 32 + 64) // 2 #56
        X = np.zeros((batch_size, 2, h, w + padding * 2), dtype=np.uint8) #1,2,2048,2560

        prev_tr = self.estimate_transformation_full(prev_img=prev_image, img=image)

        ## 仿射变换
        prev_img_aligned = cv2.warpAffine(
            prev_image,
            prev_tr[:2, :],
            dsize=(w, h),
            flags=cv2.INTER_LINEAR)
        
        # debug: 对比前后变化
        # fig, axes = plt.subplots(1, 2, figsize=(20, 20))
        # axes[0].imshow(prev_image, cmap='gray')  # 直接显示灰度图
        # axes[1].imshow(prev_img_aligned, cmap='gray')  # 显示正确颜色的方法
        # plt.show()
        
        X[0, 0, :, padding:-padding] = prev_img_aligned  # 利用前一帧转换的帧
        X[0, 1, :, padding:-padding] = image  # 当前帧

        detected_objects = predict_ensemble.predict_ensemble(
            X=X,
            models_full_res=[self.model_seg1, self.model_seg2],
            models_crops=[self.model_seg3, self.model_dla],
            full_res_threshold=0.35,
            x_offset=-padding
        )

        return detected_objects, prev_tr


class SegTracker:
    def __init__(self, detector: SegDetector):
        self.detector = detector

    def predict(self, image, prev_image):
        detected_objects, prev_tr = self.detector.detect_objects(image=image, prev_image=prev_image)

        res = []
        for detected_object in detected_objects:
            if detected_object['distance'] > 800:
                continue
            conf = detected_object['conf']
            if conf < 0.75:
                continue

            detected_object['track_id'] = len(res) + 1
            res.append(detected_object)

        return res


class SegTrackerFromOffset:
    def __init__(self, detector: SegDetector):
        self.detector = detector
        self.tracker = tracking.SimpleOffsetTracker(
            min_track_size=8,
            threshold_to_find=0.6,
            threshold_to_continue=0.6,
            threshold_distance=40,
            min_distance=1000
        )

    def predict(self, image, prev_image):
        detected_objects, prev_tr = self.detector.detect_objects(image=image, prev_image=prev_image)

        ## 将每一个检测结果存储为DetectedItem格式，放入detected_items变量，组成list
        detected_items = [
            DetectedItem(
                cx=it['cx'] + it['offset'][0],
                cy=it['cy'] + it['offset'][1],
                w=it['w'],
                h=it['h'],
                distance=it['distance'],
                confidence=it['conf'],
                track_id=-1,
                dx=it['tracking'][0],
                dy=it['tracking'][1],
                item_id=''
            )
            for it in detected_objects
            if it['conf'] >= self.tracker.threshold_to_continue
        ]

        # self.tracker 是 SimpleOffsetTracker
        items_to_submit = self.tracker.process_frame_detections(detected_items, prev_tr)

        ## 获取最终结果
        res = [
            dict(
                cx=it.cx,
                cy=it.cy,
                w=it.w,
                h=it.h,
                track_id=it.track_id,
                conf=it.confidence,
                distance=it.distance, # 添加distance
                offset=[0.0, 0.0]
            )
            for it in items_to_submit
        ]

        return res
