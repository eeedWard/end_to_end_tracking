import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from tracker_openCV import Tracker


def detect(save_txt=False, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    # classes_selected = ["bicycle", "car", "motorcycle", "airplane", "bus", "truck", "boat"]
    classes_selected = ["car", "bus", "truck"]
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    classes_selected_id_list = []
    for class_id, class_name in enumerate(classes):
        if class_name in classes_selected:
            classes_selected_id_list.append(class_id)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    tracker = Tracker(opt.tracker).tracker
    tracker_initialized = False
    update_successful = False
    tracker_bbox_xyxy = np.array([0, 0, 0, 0])
    frame_number = 0
    for path, img, im0s, vid_cap in dataset:
        frame_number+=1
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = model(img)

        if opt.half:
            pred = pred.float()

        for i, det in enumerate(non_max_suppression(pred, classes_selected_id_list, opt.conf_thres, opt.nms_thres, device)):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            update_tracker = True
            if det is not None and len(det):
                update_tracker = False

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # new detection arrived: create a new tracker
                tracker_bbox_xyxy = select_most_central_bbox(det, im0.shape).round()
                tracker = Tracker(opt.tracker).tracker
                tracker_bbox_xywh_tuple = (tracker_bbox_xyxy[0], tracker_bbox_xyxy[1], tracker_bbox_xyxy[2] - tracker_bbox_xyxy[0], tracker_bbox_xyxy[3] - tracker_bbox_xyxy[1])
                tracker_initialized = tracker.init(im0, tracker_bbox_xywh_tuple)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            if tracker_initialized and update_tracker:
                t0_tracker_update = time.time()
                update_successful, tracker_bbox_xywh_tuple = tracker.update(im0)
                tracker_processing_time = time.time() - t0_tracker_update
                print("tracker_processing_time: ", tracker_processing_time)
                tracker_bbox_xyxy = np.array([tracker_bbox_xywh_tuple[0], tracker_bbox_xywh_tuple[1], tracker_bbox_xywh_tuple[0] + tracker_bbox_xywh_tuple[2], tracker_bbox_xywh_tuple[1] + tracker_bbox_xywh_tuple[3]])

            if save_img or view_img:
                plot_bbox_label(im0, det, classes, tracker_bbox_xyxy, colors, update_successful)

            write_results(det, save_path, save_txt)

        if view_img:
            cv2.imshow(path, im0s)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

        print('%sDone with this frame in %.3fs' % (s, time.time() - t))

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)

    print()
    print('Done with all frames in %.3fs' % (time.time() - t0))


def write_results(det, save_path, save_txt=False):
    if det is not None and save_txt:
        for *xyxy, conf, _, cls in det:
            with open(save_path + '.txt', 'a') as file:
                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
def plot_bbox_label(img, det, classes, tracker_bbox, colors, update_tracker):
    if det is not None:
        for *xyxy, conf, _, cls in det:
            label = '%s %.2f' % (classes[int(cls)], conf)
            plot_one_box(xyxy, img, label=label, color=colors[int(cls)])
    if update_tracker:
        plot_one_box(tracker_bbox, img, color=(255,255,255))

def select_most_central_bbox(det, img_size):
    if det.shape[0] == 1:
        #there is only one bbox
        return det[0, :4]
    most_central_bbox = np.array([0.0, 0.0, 15.0, 30.0])
    min_distance = (img_size[0]/2.0)**2 + (img_size[1]/2.0)**2
    for bbox in det[:, :4]:
        bbox_centre = ((bbox[2] - bbox[0])/2.0, (bbox[3] - bbox[1])/2.0)
        distance = (bbox_centre[0] - (img_size[0]/2.0))**2 + (bbox_centre[1] - (img_size[1]/2.0))**2
        if distance < min_distance:
            min_distance = distance
            most_central_bbox = bbox
    return most_central_bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0,1', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--tracker', default='KCF', help='type of openCV tracker to use')
    opt = parser.parse_args()
    print()
    print(opt)
    print()

    with torch.no_grad():
        detect()
