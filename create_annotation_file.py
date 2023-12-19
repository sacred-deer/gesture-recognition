import os, argparse
import pandas as pd

def remove_non_video_files(directory_contents):
    for file in directory_contents:
        if file[-3:] not in ['mov','mp4','avi']:
            #print("File removed: ", file)
            directory_contents.remove(file)
    return directory_contents

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True, help='Path to root directory which contains the video folders for each class, containing video files.')
args = parser.parse_args()

output_file_name = 'annotations.csv'
# classes = [entry.name for entry in os.scandir(args.root_dir) if entry.is_dir()]
# if not classes:
#     raise FileNotFoundError(f"Couldn't find any class folder in {args.root_dir}.")

# class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
classes = ['hand_waving', 'pointing', 'other']
class_to_idx = {
    'hand_waving': 0,
    'pointing': 1,
    'other': 2
}

video_files_paths = []
labels = []
for class_folder in classes:
    directory_contents = os.listdir(os.path.join(args.root_dir,class_folder))
    directory_contents = remove_non_video_files(directory_contents)
    print("Number of Files detected in {}: {} || {}".format(class_folder, len(directory_contents), directory_contents))
    video_files_paths.extend([os.path.join(class_folder,video_file) for video_file in directory_contents])
    labels.extend([class_to_idx[class_folder]] * len(directory_contents))

df = pd.DataFrame.from_dict({'path':video_files_paths, 'label':labels})

df.to_csv(os.path.join(args.root_dir,output_file_name),index=False)