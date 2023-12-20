# gesture-recognition
Gesture Recognition using Person Keypoint Detection and Transfer Learning for Social Robots

## Abstract
In the realm of human interaction, non-verbal communication stands as an indispensable facet, playing a pivotal role in conveying emotions, intentions, and establishing meaningful connections. And, within the vast spectrum of non-verbal communication, gestures emerge as a fundamental and universal language. Further, in the context of social robotics, gestures play a central role in bridging the gap between humans and machines, allowing for more natural and intuitive interactions. As social robots increasingly integrate into our lives, the significance of accurate and context-aware gesture recognition becomes undeniable. In this project, our objective was to develop a gesture recognition system with the aim of enabling non-verbal communication in social robots using gestures. We have explored two approaches and investigated their efficacy in recognizing gestures. The first approach involves detecting person keypoints in the video frames, extracting engineered features, and then training a classifier on the feature dataset. The second approach is to build an end-to-end deep learning model, eliminating the need for feature engineering. Our results demonstrate that both approaches delivered promising performance and provided sufficient proof for their efficacy in recognizing gestures.


## Setting up the Local environment
### Prerequisites
* This project was built using Python 3.8.18.
### Steps:
1. [Optional] Create a python virtual environment with Python 3.8.18
2. Open Terminal and navigate inside the folder of this project
3. Run the following command to install all the required python packages

```
pip install -r requirements.txt
```
## Assumptions and Guidelines for Dataset Creation


1. There is only a single person in the video clip.
2. The gestures are performed facing straight toward the camera.
3. The hand-waving gesture is performed by oscillating the complete arm pivoted at the elbow joint, not waving just the wrist.
4. At least, the complete upper body is visible during the full length of the video clip and it does not go out of frame during the full length of the video clip.
5. The distance between the subject and the camera should be within approximately 10 feet.
6. **The video clip must contain at least 150 frames, preferably duration of 5 seconds @ 30 frames per second (fps).**
7. The complexity of the ``other" class was limited to rested poses.

## (Second Approach) Gesture Recognition with an end-to-end deep learning model

To train and test the deep learning model, the first step is to generate the annotations for our dataset. The annotations file would be created automatically by running the script ```create_annotation_file.py```. However, this script requires the dataset to be stored in the following directory structure. 
```
├── dataset
│   ├── train
│   │   ├── hand_waving
|   |   ├── pointing
|   |   ├── other
│   ├── test
│   │   ├── hand_waving
|   |   ├── pointing
|   |   ├── other

```
The main directory ("dataset") can have any folder name, but the folders inside the root directory should have the exact structure and the exact folder names.

### Generate annotation file
To create an annotation file, we will run ```create_annotation_file.py```

```
python3 create_annotation_file.py --root-dir
```
The ```--root-dir``` argument needs the path of the ```train``` or ```test``` folder. Since we need an annotations file for both train and test, we would need to run this script two times: once for train and once for test. This script would generate ```annotation.csv``` in the train and test folder, containing a list of all gesture video files and their corresponding labels. Following is the mapping of the labels:
```
{
    'hand_waving': 0,
    'pointing': 1,
    'other': 2
}
```
### Train and test the deep learning model

The Jupyter notebook ```gesture_recognition_with_few_shot_learning.ipynb``` is divided into three main sections:
1. Train the deep learning model: This section trains the deep learning model from scratch.
2. Test the deep learning model: This section loads the trained model and tests its performance on the test dataset
3. Getting a prediction on a gesture video: This section loads the trained model and generates a prediction on a given gesture video

[Note: If you want to run only one section of the notebook, make sure to execute the code before "Training the model" section because those initializations are required for all the sections]

```gesture_recognition_with_few_shot_learning.ipynb``` contains the comments inside on how to use this notebook. Please note that one will need to set the directory path variables according to their system, otherwise the code will throw an error. You will need to update the following directory paths in the code:
- Path to the dataset
- Location to save the trained model
- Location of saved model
- Path to the gesture video file for generating the prediction


### Getting a prediction on a gesture video

```gesture_recognition_with_few_shot_learning.ipynb``` contains a section titled "Getting model prediction on a gesture video". This section loads a trained model and predicts the gesture performed in the given gesture video. [Note: If you want to run only this section of the notebook, make sure to execute the code before "Training the model" section because those initializations are required for all the sections]

If one doesn't want to train the model from scratch, the latest version of the trained deep learning model ```gesture_model_v3.pt``` is provided in the ```trained_models``` folder to utilize for prediction.


## (First Approach) Person Keypoint Detection + Gesture Classifier

In this approach, we first need to detect keypoints in a gesture video and then train a gesture classifier or use a trained gesture classifier for prediction. Therefore, the first step is to detect keypoints of a person in the gesture video

### Generating Keypoints files

```gestures_to_keypoints.py``` python script processes the gesture video files and saves the keypoints of each video in a separate CSV file. This script also saves the images marked with extracted keypoints.

```
python3 gestures_to_keypoints.py --input_path --steps_between_frames --save_path
```
This python script takes 4 arguments:
1. ```--input_path```: Specify the path to the root directory of the dataset folder i.e. the path of the parent folder which contains train and test folders, according to the directory structure explained above.
2. ```--steps_between_frame```: (default = 3) Frames are sampled at regular intervals from the video. This parameter defines the interval between two samples while sampling the frames from the video. [Note: Increase the value of this argument to extract fewer frames (decreases processing time, but might reduce gesture detection accuracy).]
3. ```--save_path'```: (default = ".") Specify the directory to save the CSV files containing keypoints.

This script saves the separate CSV files containing keypoints for each video.

### Dataset of Extracted Keypoints
We have provided a dataset of person keypoints extracted from ~ 300 gesture videos. The folder ```keypoints``` contains the dataset, split into train and test. These keypoints can be used directly to train and test the gesture classifier using the ```Gesture Detection - Feature Engineering - Training.ipynb``` jupyter notebook. 

### Training and evaluating the Gesture Classifier
The Jupyter notebook ```Gesture Detection - Feature Engineering - Training.ipynb``` contains the code segments to perform the following:
1. Train the gesture classifier from scratch using the extracted keypoints
2. Test the performance of the trained gesture classifier on the test dataset

Please note that one will need to set the directory path variables according to their system, otherwise the code will throw an error. You will need to update the following directory paths in the code:
- Path to the keypoints dataset (the path to the ```keypoints``` folder which was created after running the ```gestures_to_keypoints.py``` python script)
- Location to save the trained model
- Location of the saved model (for the evaluation phase)

### Getting a prediction on a gesture video
The Jupyter notebook ```first_approach_demo.ipynb``` is created to get a prediction on gesture video using a pre-trained machine learning model (gesture classifier). The user only needs to update the following paths in the notebook:
1. Path to the gesture video
2. Path to the trained model

NOTE: DON'T RUN THIS NOTEBOOK ON GOOGLE COLAB. On colab, the python kernel would crash because of insufficient memory. The detection model requires more memory than provided by the free version of colab.

If one doesn't want to train the model from scratch, the latest version of the trained gesture classifier ```feature_engineering_model.sklearn-1-2-2.pickle``` is provided in the ```trained_models``` folder to utilize for prediction.
