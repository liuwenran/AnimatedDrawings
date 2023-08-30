from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose.util import HWC3, resize_image, draw_bodypose, draw_handpose, draw_facepose
from controlnet_aux.open_pose.hand import handDetect
from controlnet_aux.open_pose.face import faceDetect
import PIL.Image as Image
import numpy as np
import torch
import cv2
import yaml
import os

os.environ['CURL_CA_BUNDLE'] = ''


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = draw_handpose(canvas, hands)

    if draw_face:
        canvas = draw_facepose(canvas, faces)

    return canvas


    
class OpenposeDetectorPoint(OpenposeDetector):

    def __call__(self, input_image, image_path=None, detect_resolution=512, image_resolution=512, hand_and_face=False, return_pil=True):
        # hand = False
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        H_, W_, C = input_image.shape
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        input_image = input_image[:, :, ::-1].copy()
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(input_image)
            if len(candidate) == 0:
                raise "Not able to estimate pose by openpose !!!"
            hands = []
            faces = []
            if hand_and_face:
                # Hand
                hands_list = handDetect(candidate, subset, input_image)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(input_image[y:y+w, x:x+w, :]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
                # Face
                faces_list = faceDetect(candidate, subset, input_image)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(input_image[y:y+w, x:x+w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())

            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)

            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            canvas = draw_pose(pose, H, W)

        detected_map = HWC3(canvas)
        # img = resize_image(input_image, image_resolution)
        # H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W_, H_), interpolation=cv2.INTER_NEAREST)

        if return_pil:
            detected_map = Image.fromarray(detected_map)

        return detected_map, candidate, subset


def humanart(img, pose_dir):
    from mmcv.image import imread
    from mmpose.apis import inference_topdown, init_model
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples
    
    config = 'mmpose/configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w32_8xb64-210e_humanart-256x192.py'
    checkpoint = 'weights/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.pth'
    
    model = init_model(
        config,
        checkpoint,
        device='cuda:0',
        cfg_options=None)

    # init visualizer
    model.cfg.visualizer.radius = 5
    model.cfg.visualizer.alpha = 0.
    model.cfg.visualizer.line_width = 3

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
       model.dataset_meta)

    # inference a single image
    batch_results = inference_topdown(model, img)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(img, channel_order='rgb')
    H, W, C = img.shape
    
    #background = np.zeros((H, W, 3), dtype=np.int8)
    #visualizer.add_datasample(
    #    'result',
    #    background,
    #    data_sample=results,
    #    draw_gt=False,
    #    draw_bbox=True,
    #    draw_heatmap=False,
    #    show=False,
    #    out_file=pose_dir)

    # if pose_dir is not None:
    #   print_log(
    #        f'the output image has been saved at {pose_dir}',
    #        logger='current',
    #        level=logging.INFO)

    instances = results.pred_instances
    keypoints = instances.get('transformed_keypoints', instances.keypoints)
    scores = instances.keypoint_scores
    keypoints_visible = instances.keypoints_visible
    
    # convert mmpose to openpose
    keypoints_info = np.concatenate((keypoints, 
                                     scores[..., None], 
                                     keypoints_visible[...,None]),axis=-1)
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    keypoints, scores, keypoints_visible = keypoints_info[..., :2], keypoints_info[..., 2], keypoints_info[..., 3]
    
    candidate = keypoints[0]
    subset = [[i for i in range(len(candidate))] + [1] + [len(candidate)]]
    candidate[:, 0] /= float(W)
    candidate[:, 1] /= float(H)
    
    hands = []
    faces = []
    bodies = dict(candidate=candidate, subset=subset)
    pose = dict(bodies=bodies, hands=hands, faces=faces)

    # draw pose.png
    canvas = draw_pose(pose, H, W)

    detected_map = HWC3(canvas)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    detected_map = Image.fromarray(detected_map)
    detected_map.save(pose_dir)
    
    return candidate
    

def generate_pose(image_path = '../outputs/segment/image_resized.jpg',
                  mask_path = '../cutputs/segement/mask_out.jpg',
                  char_root_dir='../configs/animated_drawing/characters/',
                  char_name='tempchar',
                  pose_name = 'pose.png',
                  mode='humanart'):
    
    body_point_name = [
        # 'root',
        'hip',
        'torso',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_hand',
        'left_shoulder',
        'left_elbow',
        'left_hand',
        'right_hip',
        'right_knee',
        'right_foot',
        'left_hip',
        'left_knee',
        'left_foot',
    ]

    body_point_parent_name = [
        # 'null',
        'root',
        'hip',
        'torso',
        'torso',
        'right_shoulder',
        'right_elbow',
        'torso',
        'left_shoulder',
        'left_elbow',
        'root',
        'right_hip',
        'right_knee',
        'root',
        'left_hip',
        'left_knee',
    ]

    body_point_index = {
        'root': [8, 11],
        'hip': [8, 11],
        'torso': 1,
        'neck': 0,
        'right_shoulder': 2,
        'right_elbow': 3,
        'right_hand': 4,
        'left_shoulder': 5,
        'left_elbow': 6,
        'left_hand': 7,
        'right_hip': 8,
        'right_knee': 9,
        'right_foot': 10,
        'left_hip': 11,
        'left_knee': 12,
        'left_foot': 13,
    }

    # detect_resolution = 512
    image = Image.open(image_path)
    pose_dir = os.path.join(char_root_dir, char_name, pose_name)
    
    if mode == 'humanart':
        candidate = humanart(image_path, pose_dir)
        detected_map = Image.open(pose_dir)
        
    elif mode == 'controlnet_aux':
        control_detector = 'lllyasviel/ControlNet'
        posedet = OpenposeDetectorPoint.from_pretrained(control_detector)
        detected_map, candidate, subset = posedet(image, image_path)
        detected_map.save(pose_dir)
        
    else:
        NotImplementedError

    # resize image 
    image_np = np.array(image, dtype=np.uint8)
    image_np = HWC3(image_np)
    # image_np = resize_image(image_np, detect_resolution)
    image_resized = Image.fromarray(image_np)
    image_resized.save(os.path.join(char_root_dir, char_name, 'texture.png'))

    # resize mask image
    mask = Image.open(mask_path)
    mask_np = np.array(mask, dtype=np.uint8)
    mask_np = HWC3(mask_np)
    # mask_np = resize_image(mask_np, detect_resolution)
    image_resized = Image.fromarray(mask_np)
    image_resized.save(os.path.join(char_root_dir, char_name, 'mask.png'))


    point_location = {}
    W, H = image_resized.size
    key_list = list(body_point_index.keys())

    
    for i in range(len(key_list)):
        key = key_list[i]
        index = body_point_index[key]
        if type(index) is list:
            point_left = candidate[index[0]]
            point_right = candidate[index[1]]
            point = [(point_left[0] + point_right[0]) / 2,
                    (point_left[1] + point_right[1]) / 2]
        else:
            point = candidate[index]
        point = [int(point[0] * W), int(point[1] * H)]
        point_location[key] = point


    config_file = os.path.join(char_root_dir, char_name, 'char_cfg.yaml')
    config_dict = {}
    config_dict['width'] = detected_map.size[0]
    config_dict['height'] = detected_map.size[1]
    config_dict['skeleton'] = []

    first_item = {}
    first_item['loc'] = point_location['root']
    first_item['name'] = 'root'
    first_item['parent'] = None
    config_dict['skeleton'].append(first_item)

    for i, name in enumerate(body_point_name):
        item = {}
        item['loc'] = point_location[name]
        item['name'] = name
        item['parent'] = body_point_parent_name[i]

        config_dict['skeleton'].append(item)

    #print(config_file)
    with open(config_file, 'w') as file:
        documents = yaml.dump(config_dict, file)
    
    return pose_dir

if __name__ == '__main__':
    generate_pose(char_name='tempchar')