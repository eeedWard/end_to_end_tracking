#!/usr/bin/env python3
import rospy
import os
import cv2
import torch
import pylib.utils
import yolov3.yolov3 as yolov3

# output in a txt file in the form of:
# [(xyxy), (xyxy) .. ]--Mthesis/database/my_database/datasets/predictor_inference_fe_ds/seq_009/...

def go_through_dataset(data_dir):


    # for a folder structure like Mthesis/database/my_database/datasets/predictor_inference_fe_ds/seq_009/imgs/******.png

    for root, dirs, files in os.walk(data_dir):
        for dir in sorted(dirs): # seq001, seq002..
            for root, dirs, files in os.walk(data_dir + '/' + dir + '/imgs'):
                # root: Mthesis/database/my_database/datasets/predictor_inference_fe_ds/seq_009/imgs
                path_list = []
                boxes_list = [] # in the form [ [(xyxy), (xyxy) ..], [(xyxy), (xyxy) ..]  ]
                for file in sorted(files): #img_001.png , img_002.png...
                    if "heat_map" not in file and "target" not in file and "ss" not in file:
                        file_path = root + '/' + file
                        img = cv2.imread(file_path)
                        boxes = detect(img)
                        boxes_list.append(boxes)
                        path_list.append(file_path)
                save_box_list(boxes_list, path_list, data_dir + '/' + dir)
                rospy.loginfo("detecting imgs in " + dir)
        break #stop awfter first outer iter

    # # for a folder structure like Mthesis/database/my_database/datasets/predictor_inference_fe_ds/imgs/******.png
    # for root, dirs, files in os.walk(data_dir + '/imgs'):
    #     path_list = []
    #     boxes_list = [] # in the form [ [(xyxy), (xyxy) ..], [(xyxy), (xyxy) ..]  ]
    #     for file in sorted(files): #img_001.png , img_002.png...
    #         if "anchor"in file:
    #             file_path = root + '/' + file
    #             img = cv2.imread(file_path)
    #             boxes = detect(img)
    #             boxes_list.append(boxes)
    #             path_list.append(file_path)
    #             rospy.loginfo("Done with {}".format(file_path), end='\r')
    #     save_box_list(boxes_list, path_list, data_dir)
    #     break # loop should stop anyway

    rospy.loginfo("DONE")

def detect(img):
    with torch.no_grad():
        # (x1, y1, x2, y2, object_conf, class_conf, class)
        detections = yolov3.detect(model, img, conf_thres=0.3, nms_thres=0.5)

    boxes = thresh_normalize_boxes(detections, img.shape)
    return boxes


def thresh_normalize_boxes(r, image_shape):
    list_box = []

    im_w = float(image_shape[1])
    im_h = float(image_shape[0])

    if r is not None:
        for detection in r:
            xmin = pylib.utils.cast_01(detection[0].item() / im_w)
            ymin = pylib.utils.cast_01(detection[1].item() / im_h)
            xmax = pylib.utils.cast_01(detection[2].item() / im_w)
            ymax = pylib.utils.cast_01(detection[3].item() / im_h)

            list_box.append((xmin, ymin, xmax, ymax))

    return list_box

def save_box_list(boxes_list, path_list, save_path):
    with open(save_path + "/detector_boxes.txt", 'w') as filehandle:
        for boxes, path in zip(boxes_list, path_list):
            for box in boxes:
                for coord in box:
                    filehandle.writelines("%s, "%coord)
            filehandle.writelines("--%s\n" %path)

if __name__ == '__main__':
    rospy.init_node('detector_node')

    model_name = rospy.get_param('~model_name')

    TEST_CONFIGS = os.path.dirname(os.path.realpath(__file__)) + '/cfg/yolov3_ua/' + model_name + '.cfg'
    TEST_WEIGHTS = '/data_files/yolo_weights/' + model_name + '.weights'
    model = yolov3.load_model(TEST_CONFIGS, TEST_WEIGHTS, img_size=608)
    model = model.eval()

    data_dir = "Mthesis/database/my_database/datasets/encoder_val_ds"
    go_through_dataset(data_dir)

    rospy.spin()
    