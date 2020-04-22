#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
import cv2
import random
from yolov3.models import load_darknet_weights, Darknet
import yolov3.utils as utils
import yolov3.torch_utils as torch_utils
import yolov3.parse_config as parse_config
import torch


def load_model(config, weights, img_size=608):
    device = torch_utils.select_device("0")
    # Initialize model
    model = Darknet(config, img_size)
    _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()
    # half weights
    model.half()

    # Get classes and colors
    rospack = rospkg.RosPack()
    object_detector_path = rospack.get_path('object_detector')
    classes = utils.load_classes(object_detector_path + "/src/cfg/yolov3_ua/" + parse_config.parse_data_cfg(object_detector_path + "/src/cfg/yolov3_ua/coco.data")['names'])
    classes_selected = ["car", "bus", "truck", "boat", "cell phone"]
    classes_selected_id_list = []
    for class_id, class_name in enumerate(classes):
        if class_name in classes_selected:
            classes_selected_id_list.append(class_id)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    model.classes = classes
    model.colors = colors
    model.img_size = img_size
    model.device = device
    model.classes_ids = classes_selected_id_list

    return model

def resize_img(im0, img_size):
    # Padded resize
    img = letterbox(im0, new_shape=img_size)

    torch_img = img.transpose(2, 0, 1)
    torch_img = np.ascontiguousarray(torch_img, dtype=np.float16)  # uint8 to fp16/fp32
    torch_img /= 255.0  # normalize: 0 - 255 to 0.0 - 1.0

    return torch_img

def letterbox(img, new_shape=608, color=(128, 128, 128), mode='auto', interp=cv2.INTER_LINEAR):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        r = float(new_shape) / max(shape)  # ratio  = new / old
    else:
        r = max(new_shape) / max(shape)
    ratio = r, r  # width, height ratios
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratio = new_shape / shape[1], new_shape / shape[0]  # width, height ratios

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

def detect(model, im0, conf_thres, nms_thres):
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    img = resize_img(im0, model.img_size)
    img = torch.from_numpy(img).to(model.device)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred, _ = model(img)
    pred = pred.float()

    for i, det in enumerate(utils.non_max_suppression(pred, model.classes_ids, conf_thres, nms_thres, model.device)):  # detections per image, TODO: this loop runs only once, it is useless
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = utils.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    return det