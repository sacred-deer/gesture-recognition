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

## (Approach 2) Gesture Recognition with an end-to-end deep learning model

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

```gesture_recognition_with_few_shot_learning.ipynb``` contains the comments inside on how to use this notebook. Please note that one will need to set the directory path variables according to their system, otherwise the code will throw an error. You will need to update the following directory paths in the code:
- Path to the dataset
- Location to save the trained model
- Location of saved model
- Path to the gesture video file for generating the prediction


### Getting a prediction on a gesture video

```gesture_recognition_with_few_shot_learning.ipynb``` contains a section titled "Getting model prediction on a gesture video". This section loads a trained model and predicts the gesture performed in the given gesture video. If one doesn't want to train the model from scratch, the latest version of the trained deep learning model ```gesture_model_v3.pt``` is provided in the ```trained_models``` folder to utilize for prediction.


