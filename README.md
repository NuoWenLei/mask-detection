# mask-detection

## Goal
- "Look" at everyone's face in a photo and check if they are wearing a mask

## Web App Link
* https://mask-detector-web.herokuapp.com/

## Dataset
- Two datasets both sourced from Kaggle
  > COVID Face Mask Detection Dataset
    * https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset
  > Face Mask Detection
    * https://www.kaggle.com/andrewmvd/face-mask-detection

## Steps
- Identify all faces in image
  > Use MTCNN (Multi-Task Convolutional Neural Network)
    * Learn more at: https://github.com/ipazc/mtcnn
- Run individual classification on each face
  > Use Transfer Learning with Keras' Xception
    * https://keras.io/api/applications/xception/
- Draw box on faces with PIL

