from dataclasses import dataclass
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import common_utils
import config
import seg_prediction_to_items
import matplotlib.pyplot as plt


@dataclass
class PredictionCrop:
    crop_offset_x: int
    crop_offset_y: int
    X: torch.tensor
    pred_masks: List[np.ndarray]
    detected_object: Dict


def predict_ensemble(X, models_full_res, models_crops, full_res_threshold, x_offset):
    predictions_full_res = []
    torch.set_grad_enabled(False)

    X = torch.from_numpy(X).cuda().float() / 255.0  #1,2,2048,2560

    ## models_full_res： HRNetSeg 和 EffDetSegmentation8x 两个模型分别保存推理结果
    for model in models_full_res:
        with torch.cuda.amp.autocast():
            pred = model(X)

        pred['mask'] = torch.sigmoid(pred['mask'][0].float()) # 1,1,256,320
        for k in list(pred.keys()):
            pred[k] = pred[k][0].float().cpu().detach().numpy()

        predictions_full_res.append(pred)

    predictions_masks_full_res = [pred['mask'] for pred in predictions_full_res] # list，256,320
    ## debug: 查看mask
    # fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    # axes[0].imshow(predictions_masks_full_res[0], cmap='gray')
    # axes[1].imshow(predictions_masks_full_res[1], cmap='gray')
    # plt.show()
    comb_pred = np.mean(predictions_masks_full_res, axis=0)

    # mask borders, known not to have images or very close to borders,
    # it's more likely prediction would not match annotations
    border_x = -x_offset // 8 + 4  # 11
    border_y = 4
    # print(x_offset, border_x, border_y) #-56， 11， 4
    # plt.imshow(comb_pred)
    # plt.figure()
    comb_pred[:border_y, :] = 0.0
    comb_pred[-border_y-4:, :] = 0.0
    comb_pred[:, :border_x] = 0.0
    comb_pred[:, -border_x:] = 0.0
    # plt.imshow(comb_pred)
    # plt.show()

    pred = predictions_full_res[0]  # 取HRNetSeg 结果
    ## pred_to_items会计算出distance
    detected_objects_full_res = seg_prediction_to_items.pred_to_items(
        comb_pred=comb_pred,
        offset=pred['offset'],
        size=pred['size'],
        tracking=pred['tracking'],
        distance=pred['distance'][0],
        above_horizon=pred['above_horizon'][0],
        conf_threshold=full_res_threshold,
        pred_scale=8,
        x_offset=x_offset
    )
    
    ## 若没有追踪到，则直接返回
    if len(detected_objects_full_res) == 0 or len(models_crops) == 0:
        return detected_objects_full_res

    # 最去取多少物体
    max_nb_objects = 6
    detected_objects_full_res = detected_objects_full_res[:max_nb_objects]

    h, w = X.shape[-2:] # 2048,2560
    crop_size = 512

    crops = []
    
    ## 裁切图片后，送入models_crops模型 model_seg3, model_dla
    for obj in detected_objects_full_res:
        cx = int(np.clip(obj['cx'], crop_size / 2, w - crop_size / 2))
        cy = int(np.clip(obj['cy'], crop_size / 2, h - crop_size / 2))

        crop_offset_x = (cx - crop_size // 2) // 32 * 32
        crop_offset_y = (cy - crop_size // 2) // 32 * 32

        X_crop = X[:, :, crop_offset_y:crop_offset_y+crop_size, crop_offset_x:crop_offset_x+crop_size] # (1,2, 512, 512)
        crops.append(PredictionCrop(X=X_crop, crop_offset_x=crop_offset_x, crop_offset_y=crop_offset_y, detected_object=obj, pred_masks=[]))

    X_crop_combined = torch.cat([c.X for c in crops], dim=0)
    for model in models_crops:
        with torch.cuda.amp.autocast():
            pred = model(X_crop_combined)

        pred_mask = torch.sigmoid(pred['mask'].float()).cpu().detach().numpy()
        for i, crop in enumerate(crops):
            crop.pred_masks.append(pred_mask[i, 0])
        del pred

    res = []

    for i, crop in enumerate(crops):
        crop_offset_x = crop.crop_offset_x
        crop_offset_y = crop.crop_offset_y

        ## 全图预测结果缩放 + 裁剪图预测
        predictions_masks_crops = [
            p[crop_offset_y//8: (crop_offset_y+crop_size)//8, crop_offset_x//8: (crop_offset_x+crop_size)//8]
            for p in predictions_masks_full_res
        ] + crop.pred_masks

        # fig, ax = plt.subplots(len(predictions_masks_crops))
        # for i, p in enumerate(predictions_masks_crops):
        #     ax[i].imshow(p, vmin=0, vmax=1)
        # plt.figure()

        comb_pred_crop = np.mean(predictions_masks_crops, axis=0)
        # pos_max = common_utils.argmax2d(comb_pred_crop)
        crop_x = np.round((crop.detected_object['cx'] - crop_offset_x - x_offset) / 8 - 0.5)
        crop_y = np.round((crop.detected_object['cy'] - crop_offset_y) / 8 - 0.5)

        crop_x = int(np.clip(crop_x, 1, comb_pred_crop.shape[-1] - 2)) # 41
        crop_y = int(np.clip(crop_y, 1, comb_pred_crop.shape[-2] - 2)) # 32
        conf_updated = np.max(comb_pred_crop[crop_y-1:crop_y+2, crop_x-1:crop_x+2]) # 置信度，0.93
        # print(f"conf orig {crop.detected_object['conf']} updated {conf_updated}")
        crop.detected_object['conf'] = conf_updated
        res.append(crop.detected_object)

    return res



def check_predict_ensemble():
    part = 'part1'
    flight_id = '027b2fd1f91d4fdfbf5ceffc3c4280b7'

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    from train import build_model

    models_names = [
        ('100_hrnet32_fpm', 2220),
        # ('100_gernet_m_b2_fpm', 2220),
        ('100_gernet_m_b2_fpm', 2220),
        ('100_edet_b5_fpm', 1560),
    ]

    models = []
    for model_name, epoch in models_names:
        model = build_model(common_utils.load_config_data(model_name))
        model = model.cuda()
        model.eval()
        checkpoints_dir = f"../output/checkpoints/{model_name}/0"
        print(f"{checkpoints_dir}/{epoch:03}.pt")
        checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])

        models.append(model)

    df = pd.read_csv(f'../data/{part}/ImageSets/groundtruth.csv')
    frame_transforms = pd.read_pickle(f'{config.DATA_DIR}/frame_transforms/{part}/{flight_id}.pkl')

    df_flight = df[df.flight_id == flight_id][['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
    frame_numbers = df_flight.frame.values
    file_names = [f'{config.DATA_DIR}/{part}/Images/{flight_id}/{fn[:-4]}.jpg' for fn in df_flight.img_name]

    prev_image = None
    for i, frame_num in enumerate(tqdm(frame_numbers)):
        image = cv2.imread(file_names[i], cv2.IMREAD_GRAYSCALE)

        if prev_image is None:
            prev_image = image
            continue

        tr = frame_transforms.iloc[i]
        dx = tr['dx']
        dy = tr['dy']
        angle = tr['angle']

        batch_size = 1
        h, w = image.shape
        padding = (16 + 32 + 64) // 2
        X = np.zeros((batch_size, 2, h, w + padding * 2), dtype=np.uint8)

        prev_tr = common_utils.build_geom_transform(
            dst_w=w,
            dst_h=h,
            src_center_x=w / 2 + dx,
            src_center_y=h / 2 + dy,
            scale_x=1.0,
            scale_y=1.0,
            angle=angle,
            return_params=True
        )

        prev_img_aligned = cv2.warpAffine(
            prev_image,
            prev_tr[:2, :],  # the same crop as with the cur image
            dsize=(w, h),
            flags=cv2.INTER_LINEAR)

        X[0, 0, :, padding:-padding] = prev_img_aligned
        X[0, 1, :, padding:-padding] = image
        prev_image = image

        detected_objects = predict_ensemble(
            X=X,
            models_full_res=models[:2],
            models_crops=models[2:],
            full_res_threshold=0.5,
            x_offset=-padding
        )

        if len(detected_objects) == 0:
            continue

        # plt.imshow(image, cmap='gray')
        # ax = plt.gca()
        #
        # for res in detected_objects:
        #     confidence = float(res['conf'])
        #     cx = res['cx'] + res['offset'][0]
        #     cy = res['cy'] + res['offset'][1]
        #     w = res['w']
        #     h = res['h']
        #
        #     bbox = [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]
        #     rect = matplotlib.patches.Rectangle(bbox[:2], w, h, linewidth=1, edgecolor='b', facecolor='none')
        #     ax.add_patch(rect)
        #     plt.text(bbox[0], bbox[1], f'{int(confidence * 100)}', c='yellow')
        #
        # plt.show()


if __name__ == '__main__':
    check_predict_ensemble()












