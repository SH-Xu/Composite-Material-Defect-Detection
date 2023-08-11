## Composite Material Defect Detection
A GUI enabling recognizing and annotating the defects in the composite material backlight images using binary semantic segmentation.

# Introduction
This is an application designed for both the binary annotation and recognization of the defects in the composite material backlight images. The data annotation and result revision are combined, so that the users can use the same tools to do both binary annotation of original images, and revision of the segmentation result. The application uses U-Net for semantic segmentation. The model and the pretrained parameters are adopted from https://github.com/MitraDP/Detection-of-Surface-Defects-in-Magnetic-Tile-Images, and are trained on the train dataset ./dataset/set1. The revised results can be saved to the train dataset ./dataset/set1 to update the train set, with which the model parameters can be updated for higher accuracy.

# Configuration
All the codes are written in Python3. The GUI is implemented with PyQt5, and the deep learning code is implemented with Pytorch. Other required packages are referred to requirements.txt.

# Usage
The main file is defect_seg_GUI.py. Run this file in the above environment. Click "Segmentation" to do binary semantic segmentation of the defects. Click "Annotation" to annotate the original images. Click "Revision" to revise the segmentation results. Click "Update model" to update the model parameters with the updated train set. 
