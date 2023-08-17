# Composite Material Defect Detection
A GUI enabling recognizing and annotating the defects in the composite material backlight images using binary semantic segmentation.

## Introduction
This is an application designed for both the binary annotation and recognization of the defects in the composite material backlight images. The data annotation and result revision are combined, so that the users can use the same tools to do both binary annotation of original images, and revision of the segmentation results. The application uses U-Net for semantic segmentation. The model and the pretrained parameters are adopted from https://github.com/MitraDP/Detection-of-Surface-Defects-in-Magnetic-Tile-Images, and are trained on the self-made train dataset ./dataset/set1. The revised results can be saved to the train dataset ./dataset/set1 to update the train set, with which the model parameters can be updated for higher accuracy. The trained model is validated on ./dataset/set2.

The main purpose of this repository is to share the GUI. So to protect the privacy, only part of the self-made data is given. You can use the original dataset from https://github.com/abin24/Magnetic-tile-defect-datasets. with trained model parameter from https://github.com/MitraDP/Detection-of-Surface-Defects-in-Magnetic-Tile-Images to replace the dataset and model parameter in this repository instead. Please contact me with other queries.

## Configuration
All the codes are written in Python3.10. The GUI is implemented with PyQt5, and the deep learning code is implemented with Pytorch2.0.1. Other required packages are referred to `requirements.txt`.
```
pip install requirements.txt
```

## Usage
To use the application, run the main file `main.py` with the above configurations. Use "Open" and "Save" in "File" menu to open and save images and masks. The opened original image is shown in the left box. Click "Annotate" to annotate the original images in the right box. The widths of brush and eraser can be adjusted according to needs. Use mouse middle button to scale and drag the images.
![annotation](https://github.com/SH-Xu/Composite-Material-Defect-Detection/blob/main/example_image/annotation.png)
Click "Detect" to do binary semantic segmentation. The results are shown over the background image in the right box.
![detect](https://github.com/SH-Xu/Composite-Material-Defect-Detection/blob/main/example_image/detect.png)
Click "Revise" to revise the segmentation results.
![revise](https://github.com/SH-Xu/Composite-Material-Defect-Detection/blob/main/example_image/revise.png)
Use "Scale" to add a scale, and "Measure Length" to measure the length on the original image using the provided scale.
![measure](https://github.com/SH-Xu/Composite-Material-Defect-Detection/blob/main/example_image/measure.png)
In addition, click "Update model" to update the model parameters with the updated train set. Click "Clear" to clear both image and mask. The interface for choosing objects and algorithms are provided, but are not implemented yet.
