#loading libraries
import numpy as np
import matplotlib.pyplot as plt
import random, string, argparse, os
from pathlib import Path
import pandas as pd

import torch
import torchvision.transforms.functional as F
from torchvision.io.video import read_video
from torchvision.utils import draw_keypoints
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision.io import read_image


device = torch.device("mps")
plt.rcParams["savefig.bbox"] = 'tight'

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='Path to the dataset containing video files.')
parser.add_argument('--step_between_frames', type=int, default=3, help='This paramtere defines the interval between to samples while sampling the frames from the video. Increase to extract fewer frames (decreases processing time, but might reduce gesture detection accuracy).')
parser.add_argument('--save_path', default=".", type=str, help='Path to save the csv containing keypoints.')

args = parser.parse_args()

# checking if the path to save keypoints exists or not
if not os.path.exists(args.save_path):
        print(f"~~~ The directory '{args.save_path}' does not exist. ~~~~")
        exit(0)

if not os.path.exists(args.input_path):
        print(f"~~~ The directory '{args.input_path}' does not exist. ~~~~")
        exit(0)

#creating the directory structure for keypoints folder
os.makedirs(os.path.join(args.save_path, "keypoints", "train", "hand_waving"))
os.makedirs(os.path.join(args.save_path, "keypoints", "train", "pointing"))
os.makedirs(os.path.join(args.save_path, "keypoints", "train", "other"))

os.makedirs(os.path.join(args.save_path, "keypoints", "test", "hand_waving"))
os.makedirs(os.path.join(args.save_path, "keypoints", "test", "pointing"))
os.makedirs(os.path.join(args.save_path, "keypoints", "test", "other"))

# This threshold defines the minimum confidence score required while detecting keypoints
detect_threshold = 0.75

#defining functions and constants
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# This function removes all the non-video files from the list when reading videos from the directory
def remove_non_video_files(directory_contents):
    for file in directory_contents:
        if file[-3:] not in ['mov','mp4','avi']:
            print("Deleted file: ", file)
            directory_contents.remove(file)
    return directory_contents

nan_values = torch.tensor([float('nan'), float('nan'), float('nan')])

coco_keypoints = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

cols = []
for kp in coco_keypoints:
    cols.append(kp + "_x")
    cols.append(kp + "_y")

connect_skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]

keypoint_to_index = {
    "nose":0, "left_eye":1, "right_eye":2, "left_ear":3, "right_ear":4,
    "left_shoulder":5, "right_shoulder":6, "left_elbow":7, "right_elbow":8,
    "left_wrist":9, "right_wrist":10, "left_hip":11, "right_hip":12,
    "left_knee":13, "right_knee":14, "left_ankle":15, "right_ankle":16,
}

# grid = make_grid(img_list)
# show(grid)

#Loading the model
print("===> Loading the model")
weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)#.to(device)
model = model.eval()
print("===> The keypoint detector model loaded!")



for folder_type in ['train','test']:
    for gesture_type in ['hand_waving', 'pointing', 'other']:
        directory_path = os.path.join(args.input_path, folder_type, gesture_type)
        
        directory_contents = os.listdir(directory_path)
        #keeping only video files
        directory_contents = remove_non_video_files(directory_contents)
        #print("Video File names: ", directory_contents)

        # Loading video
        for i, video_file in enumerate(directory_contents):
            print("\t>>>>>> {}/{}: Progress: {}/{} >>>>>>".format(folder_type, gesture_type, i+1,len(directory_contents)))
            print("\n***** PROCESSING VIDEO FILE: ", video_file, " *****")
            print("===> Loading the video")
            video_loc = os.path.join(args.input_path, folder_type, gesture_type, video_file)
            frames, t1, t2 = read_video(video_loc, output_format="TCHW")
            print("Video metadata: ", t2)
            print("Total number of frames extracted: ", len(frames))
            # Fetching selected frames
            #gesture_int = [frames[x].to(device) for x in range(0,len(frames),step_between_frames)]
            gesture_int = [frames[x] for x in range(0,len(frames),args.step_between_frames)]
            print("Total number of frames selected: ", len(gesture_int))

            print("===> Preprocessing the frames for the model")
            # gestures_float = [transforms(img.to(device)) for img in gesture_int]
            gestures_float = [transforms(img) for img in gesture_int]

            # Detecting the keypoints in the frames
            print("===> Processing frames to detect keypoints")
            #with torch.no_grad():
            outputs = []
            for gesture in gestures_float:
                outputs.append(model([gesture])[0])
            #outputs = model(gestures_float)
            print("===> The video is processed")

            # Keeping the keypoints of those objects from the video whose confidence score was above threshold
            for i in range(len(outputs)):
                idx = torch.where(outputs[i]['scores'] > detect_threshold)
                #if the keypoint is not available, then making it nan
                if len(idx[0]) == 0:
                    print("====> No person detected. Skip and Go ahead! <====")
                    outputs[i]['keypoints'][0] = nan_values
                else:
                    outputs[i]['keypoints'] = outputs[i]['keypoints'][idx]
                    outputs[i]['keypoints_scores'] = outputs[i]['keypoints_scores'][idx]
                    # Also, filling nan values for keypoints whose confidence score is in negative (=> couldn't find that joint)
                    outputs[i]['keypoints'][0][torch.where(outputs[i]['keypoints_scores'][0] < 0)] = nan_values
            
            # formatting the data so that it can be saved as DataFrame csv
            data_csv = [outputs[idx]['keypoints'][0][:,0:2].detach().numpy().flatten() for idx in range(0,len(outputs))]

            df = pd.DataFrame(data=data_csv, columns=cols)
            
            to_save_path = os.path.join(args.save_path, "keypoints", folder_type, gesture_type)

            print("===> Saving the csv file containing keypoints at ", "{}/keypoints_{}.csv".format(to_save_path, "_".join(video_loc.split('/')[-1].split('.'))))
            df.to_csv("{}/keypoints_{}.csv".format(to_save_path, "_".join(video_loc.split('/')[-1].split('.'))),index=False)

            # print("===> Saving the images at ", args.path_to_keypoint_imgs)
            # for img_idx in range(len(outputs)):
            #     res = draw_keypoints(gesture_int[img_idx], outputs[img_idx]['keypoints'], colors="blue", radius=3)
            #     save_image(F.convert_image_dtype(res, dtype=torch.float32), '{}/{}_{}.png'.format(args.path_to_keypoint_imgs, "_".join(video_loc.split('/')[-1].split('.')), img_idx))








