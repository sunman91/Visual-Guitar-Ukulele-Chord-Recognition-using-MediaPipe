# Visual-Guitar-Ukulele-Chord-Recognition-using-MediaPipe
A Computer Vision project for Visual Recognition of chords from hand poses for Guitar and Ukulele using MediaPipe.

## Description 🎼
This project uses the extracted hand landmarks from images using MediaPipe to recognise chord shapes. Does not take into account fret position.
i.e. All bar chords will be considered F

Extrapolation of https://github.com/akshaybahadur21/Guitar-Learner.
- Trained with more difficult and more ambiguous guitar chord shapes
- Transfer learning on ukulele using pretrained gutiar model for chord identfication. Recognises the 'D' shape from guitar as 'G' in ukulele as they have the same shape
- Trained with ukulele dataset for accurate results

![Pipeline](https://user-images.githubusercontent.com/51843952/207555112-9cbfc75e-991f-40d1-93e7-686716f568b9.png)

Check pdf for the paper
https://github.com/sunman91/Visual-Guitar-Ukulele-Chord-Recognition-using-MediaPipe/blob/main/Visual%20Guitar%20and%20Ukulele%20Chord%20Classification%20using%20MediaPipe.pdf

## Code Requirements 🦄
You can install Conda for python which resolves all the dependencies for machine learning.

## Setup 🖥️
- Use 'CreateDataset_ukelele.py' or 'CreateDataset.py' to take image frames from webcam video to create an image dataset to train. Enter the gesture name 
- Use 'ukelele_trainer.py' or 'resnet_trainer.py' to train the model from the above created dataset

## Execution 🐉

```
python3 GuitarLearner.py
```

## Results 
![Results](https://user-images.githubusercontent.com/51843952/207555550-ba18789b-f421-403c-836d-a48949ed7da1.png)


## References 🔱
 
 -  Ivan Grishchenko and Valentin Bazarevsky, Research Engineers, Google Research. [Mediapipe by Google](https://github.com/google/mediapipe)
 -  "GitHub - akshaybahadur21/Guitar-Learner: Guitar chord detection and classifier for humans 🎸", GitHub, 2022. [Online]. Available:
    https://github.com/akshaybahadur21/Guitar-Learner.[Accessed: 24- Jul- 2022]



