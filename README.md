# Malayalam Character Recognition
Malayalam handwritten character recognition using convolutional neural networks.

## Dataset
Download the dataset from [here](https://drive.google.com/open?id=1WjZnnmmfjv7-N-WakhJdLoDhiHEi5dOb "Google Drive link") and extract the contents to `rawdata`

## Libraries used
* [Augmentor](https://github.com/mdbloice/Augmentor)
* [OpenCV](https://opencv.org/)
* [Numpy](https://numpy.org/)
* [Keras](https://keras.io/)

## How-To
Install libraries : `pip install -r requirements.txt`

Augment the data : `python dataset_augmentor.py`  
Crop and resize the images : `python data_cleaner.py`  
Process the images : `python data_process.py`  
Train the network : `python train.py`  
Predict an image : `python scan.py -i <filename>`