import sys
import os
import math

from PyQt5.QtCore import QSize, Qt, QPointF
from PyQt5.QtGui import QColor, QPainter, QPixmap, QImage, QCursor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QWidget,
    QToolBar,
    QAction,
    QStackedLayout,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QComboBox,
    QStatusBar,
    QSpinBox,
    QFrame
)

from qt_material import apply_stylesheet

import numpy as np
import pandas as pd
from PIL import Image, ImageQt
import matplotlib.pyplot as plt

import torch
import torchvision.transforms
from glob import glob
import torch.optim as optim

from unet import UNet_2D
from train import train_2D
from dataset import DefectDetectionDataset
from loss import WeightedBCELoss, TverskyLoss

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1800, 800)
        self.setWindowTitle("Composite Material Defect Detection")
        
        self.setupWindow()
        self.setupMenu()
        self.setupToolbar()

        self.detect_model = UNet_2D(1,1,32,0.2).cuda()
        # load the model
        self.detect_model.load_state_dict(torch.load('model.pt'))
    
    def setupWindow(self):
        self.total_layout = QVBoxLayout()
        # 必须先进行以下设置，否则QStackedLayout的前后顺序会出问题
        # 变成index小的层在下面（本来应该在上面）
        container = QWidget()
        container.setLayout(self.total_layout)
        self.setCentralWidget(container)

        # must set up in order
        self.setStatusLayout()
        self.setImageLayout()
        self.setInfoLayout()
        # Create container widget and set main window's widget
        
    
    def setStatusLayout(self):
        # layout to show current status, including the algorithm used and the operation being done
        self.selection_box = QHBoxLayout()
        self.total_layout.addLayout(self.selection_box)
        object_label = QLabel("Detect Object: ")
        object_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        object_box = QComboBox()
        object_box.addItem("Backlight Images")
        object_box.currentIndexChanged.connect(self.objectChanged)
        algorithm_label = QLabel("Detect Algorithm: ")
        algorithm_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        algorithm_box = QComboBox()
        algorithm_box.addItem("U-Net")
        object_box.currentIndexChanged.connect(self.algorithmChanged)
        status_label = QLabel("Current Status: ")
        status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_show = QLabel("")
        self.status_show.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.selection_box.addWidget(object_label)
        self.selection_box.addWidget(object_box)
        self.selection_box.addWidget(algorithm_label)
        self.selection_box.addWidget(algorithm_box)
        self.selection_box.addWidget(status_label)
        self.selection_box.addWidget(self.status_show)
    
    def objectChanged(self):
        # change recognizing object to that selected from object_box
        # to be implemented
        pass

    def algorithmChanged(self):
        # change algorithm to that selected from algorithm_box
        # to be implemented
        pass

    def setImageLayout(self):
        # layout to display images
        self.image_layout = QHBoxLayout()
        self.total_layout.addLayout(self.image_layout)

        vline1 = QFrame()
        vline1.setFrameShape(QFrame.VLine)
        vline2 = QFrame()
        vline2.setFrameShape(QFrame.VLine)

        # must set up in order
        self.setOriginalBox()
        self.image_layout.addWidget(vline1)
        self.setAnnotationBox()
        self.image_layout.addWidget(vline2)
        self.setToolBox()

    def setOriginalBox(self):
        # layout to display the original iamge
        original_v_box = QVBoxLayout()
        self.image_layout.addLayout(original_v_box, Qt.AlignmentFlag.AlignCenter)
        # self.image_layout.addLayout(original_v_box)

        original_img_header = QLabel("Original Image")
        original_img_header.setStyleSheet("font-weight:bold")
        original_img_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_v_box.addWidget(original_img_header)

        self.original_label = QStackedLayout()
        self.original_label.setStackingMode(QStackedLayout.StackingMode.StackAll)
        original_v_box.addLayout(self.original_label)

        self.default_back_pixmap = QPixmap(QSize(800, 600))
        self.default_back_pixmap.fill(QColor(48, 76, 98))
        self.original_layer = Background(QSize(800, 600))
        # self.original_layer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_layer.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.ruler_layer = Ruler(QSize(800, 600))
        self.ruler_layer.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.original_label.addWidget(self.ruler_layer)
        self.original_label.addWidget(self.original_layer)

        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ruler_layer.background_partner = self.original_layer
    
    def setAnnotationBox(self):
        # layout to display the annotated image
        annotation_v_box = QVBoxLayout()
        self.image_layout.addLayout(annotation_v_box, Qt.AlignmentFlag.AlignCenter)
        # self.image_layout.addLayout(annotation_v_box)

        annotation_img_header = QLabel("Segmentation Result")
        annotation_img_header.setStyleSheet("font-weight:bold")
        annotation_img_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        annotation_v_box.addWidget(annotation_img_header)

        self.annotation_label = QStackedLayout()
        self.annotation_label.setStackingMode(QStackedLayout.StackingMode.StackAll)
        annotation_v_box.addLayout(self.annotation_label)

        self.back_layer = Background(QSize(800, 600))
        # 设置成中心对齐，鼠标位置就不对了，可能要重新算过
        self.back_layer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.back_layer.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.canvas = Canvas(QSize(800, 600))
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.canvas.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # 先加入的在上面
        self.annotation_label.addWidget(self.canvas)
        self.annotation_label.addWidget(self.back_layer)
        
        self.annotation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # self.original_layer.background_partner = self.back_layer
        # self.back_layer.background_partner = self.original_layer
        # self.back_layer.canvas_partner = self.canvas
        self.canvas.background_partner = self.back_layer

    def setToolBox(self):
        # layout of tools for ruler
        tool_box = QVBoxLayout()
        self.image_layout.addLayout(tool_box, Qt.AlignmentFlag.AlignCenter)
    
        self.ruler_layer.is_annotation = False

        rulerbox_header = QLabel("Scale")
        rulerbox_header.setStyleSheet("font-weight:bold")
        rulerbox_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_box.addWidget(rulerbox_header)
        self.ruler_button = QPushButton("Add Scale")
        self.ruler_button.clicked.connect(self.rulerButtonClicked)
        self.ruler_button.setFixedSize(QSize(210, 40))
        tool_box.addWidget(self.ruler_button)
        self.ruler_label = QLabel("Scale is: ")
        self.ruler_layer.related_ruler_label = self.ruler_label
        tool_box.addWidget(self.ruler_label)

        self.measure_button = QPushButton("Measure Length")
        self.measure_button.clicked.connect(self.measureButtonClicked)
        self.measure_button.setFixedSize(QSize(210, 40))
        tool_box.addWidget(self.measure_button)
        self.measure_label = QLabel("Length is: ")
        self.ruler_layer.related_measure_label = self.measure_label
        tool_box.addWidget(self.measure_label)

        # layout of tools for annotation
        self.canvas.is_annotation = False # set the mode of whether annotating

        toolbox_header = QLabel("Annotation Tools")
        toolbox_header.setStyleSheet("font-weight:bold")
        toolbox_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_box.addWidget(toolbox_header)

        self.is_revise_button = QPushButton("Revise")
        self.is_revise_button.clicked.connect(self.reviseButtonChecked)
        self.is_revise_button.setCheckable(True)
        self.is_revise_button.setFixedSize(QSize(210, 40))
        tool_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_box.addWidget(self.is_revise_button)

        pen_box = QHBoxLayout()
        tool_box.addLayout(pen_box, Qt.AlignmentFlag.AlignCenter)

        self.pen_button = QPushButton("Brush")
        self.pen_button.clicked.connect(self.penButtonClicked)
        self.pen_button.setEnabled(self.canvas.is_annotation)
        self.pen_button.setCheckable(True)
        self.pen_button.setFixedSize(QSize(100,40))

        self.eraser_button = QPushButton("Eraser")
        self.eraser_button.clicked.connect(self.eraserButtonClicked)
        self.eraser_button.setEnabled(self.canvas.is_annotation)
        self.eraser_button.setCheckable(True)
        self.eraser_button.setFixedSize(QSize(100,40))
        
        pen_box.addWidget(self.pen_button)
        pen_box.addWidget(self.eraser_button)

        pen_width_header = QLabel("Brush Width")
        pen_width_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_box.addWidget(pen_width_header)
        self.pen_width_box = QSpinBox()
        self.pen_width_box.setRange(1, 20)
        self.pen_width_box.setValue(5)
        self.pen_width_box.setSuffix(' pix')
        self.pen_width_box.setStyleSheet("color:white")
        # pen_width_box.setFont()
        self.pen_width_box.valueChanged.connect(self.penWidthChanged)
        self.pen_width_box.setEnabled(self.canvas.is_annotation)
        self.pen_width_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_box.addWidget(self.pen_width_box)

        eraser_width_header = QLabel("Eraser Width")
        eraser_width_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_box.addWidget(eraser_width_header)
        self.eraser_width_box = QSpinBox()
        self.eraser_width_box.setRange(1, 20)
        self.eraser_width_box.setValue(10)
        self.eraser_width_box.setSuffix(' pix')
        self.eraser_width_box.setStyleSheet("color:white")
        # pen_width_box.setFont()
        self.eraser_width_box.valueChanged.connect(self.eraserWidthChanged)
        self.eraser_width_box.setEnabled(self.canvas.is_annotation)
        self.eraser_width_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_box.addWidget(self.eraser_width_box)
    
    def rulerButtonClicked(self):
        self.ruler_layer.is_annotation = True
        self.ruler_layer.is_ruler = True
        self.ruler_layer.setCursor(QCursor(Qt.CrossCursor))
    
    def measureButtonClicked(self):
        self.ruler_layer.is_annotation = True
        self.ruler_layer.is_measure = True
        self.ruler_layer.setCursor(QCursor(Qt.CrossCursor))

    def reviseButtonChecked(self, checked):
        self.canvas.is_annotation = checked
        if checked:
            self.canvas.setCursor(QCursor(Qt.CrossCursor)) # become cross for painting
            self.pen_button.setEnabled(self.canvas.is_annotation)
            self.eraser_button.setEnabled(self.canvas.is_annotation)
            self.pen_width_box.setEnabled(self.canvas.is_annotation)
            # self.eraser_width_box.setEnabled(self.canvas.is_annotation)
            self.pen_button.setChecked(True) # default painter is pen
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor)) # become cross for painting
            self.pen_button.setEnabled(self.canvas.is_annotation)
            self.eraser_button.setEnabled(self.canvas.is_annotation)
            self.pen_width_box.setEnabled(self.canvas.is_annotation)
            self.eraser_width_box.setEnabled(self.canvas.is_annotation)
            self.pen_button.setChecked(False)
            self.eraser_button.setChecked(False)

    def penButtonClicked(self):
        self.canvas.is_eraser = False
        self.pen_button.setChecked(True)
        self.eraser_button.setChecked(False) # see QAbstractButton
        self.pen_width_box.setEnabled(True)
        self.eraser_width_box.setEnabled(False)
        self.canvas.pen_width = self.pen_width_box.value()

    def eraserButtonClicked(self):
        self.canvas.is_eraser = True
        self.pen_button.setChecked(False)
        self.eraser_button.setChecked(True)
        self.pen_width_box.setEnabled(False)
        self.eraser_width_box.setEnabled(True)
        self.canvas.pen_width = self.eraser_width_box.value()
    
    def penWidthChanged(self, i):
        if not self.canvas.is_eraser:
            self.canvas.pen_width = i

    def eraserWidthChanged(self, i):
        if self.canvas.is_eraser:
            self.canvas.pen_width = i

    def setInfoLayout(self):
        # layout to show infomation, like the percentage of the defects
        pass
    
    def setupMenu(self):
        # Create menu bar
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        
        # file menu
        # =================================================
        # Create actions for file menu
        open_image_act = QAction("Open Image", self)
        open_image_act.setShortcut('Ctrl+O')
        open_image_act.setStatusTip("Open image.")
        open_image_act.triggered.connect(self.openImageFile)

        open_mask_act = QAction("Open Mask", self)
        open_mask_act.setShortcut('Ctrl+M')
        open_mask_act.setStatusTip("Open mask (after opening image).")
        open_mask_act.triggered.connect(self.openMaskFile)

        save_image_mask_act = QAction("Save Image & Mask", self)
        save_image_mask_act.setShortcut('Ctrl+S')
        save_image_mask_act.setStatusTip("Save image and mask with the same file name.")
        save_image_mask_act.triggered.connect(self.saveImageMaskFile)

        save_image_act = QAction("Save Image", self)
        save_image_act.setShortcut('Ctrl+Q')
        save_image_act.setStatusTip("Save image.")
        save_image_act.triggered.connect(self.saveImageFile)

        save_mask_act = QAction("Save Mask", self)
        save_mask_act.setShortcut('Ctrl+W')
        save_mask_act.setStatusTip("Save mask.")
        save_mask_act.triggered.connect(self.saveMaskFile)

        # Create file menu and add actions
        file_menu = menu_bar.addMenu("File")
        open_menu = file_menu.addMenu("Open...")
        open_menu.addAction(open_image_act)
        open_menu.addSeparator
        open_menu.addAction(open_mask_act)

        save_menu = file_menu.addMenu("Save...")
        save_menu.addAction(save_image_mask_act)
        save_menu.addSeparator
        save_menu.addAction(save_image_act)
        save_menu.addAction(save_mask_act)

        self.image_file = None
        self.image = None
        # image_pixmap is managed by Background
        self.mask_file = None
        self.mask_image = None
        # mask_pixmap is managed by Canvas
        # =================================================

        # operation menu
        # =================================================
        # Create actions for operqtion menu
        self.segmentation_act = QAction("Detect", self)
        self.segmentation_act.setShortcut('Ctrl+G')
        self.segmentation_act.setStatusTip("Do semantic segmentation.")
        self.segmentation_act.triggered.connect(self.doSegmentation)

        self.annotation_act = QAction("Annotate", self)
        self.annotation_act.setShortcut('Ctrl+H')
        self.annotation_act.setStatusTip("Do annotation on original images.")
        self.annotation_act.triggered.connect(self.doAnnotation)

        self.revise_act = QAction("Revise", self)
        self.revise_act.setShortcut('Ctrl+J')
        self.revise_act.setStatusTip("Do revision on segmentation results.")
        self.revise_act.triggered.connect(self.doRevise)

        self.clear_act = QAction("Clear", self)
        self.clear_act.setShortcut('Ctrl+K')
        self.clear_act.setStatusTip("Clear both image and mask.")
        self.clear_act.triggered.connect(self.doClear)

        self.retrain_act = QAction("Update model", self)
        self.retrain_act.setShortcut('Ctrl+L')
        self.retrain_act.setStatusTip("Use updated train set to update model parameters.")
        self.retrain_act.triggered.connect(self.doRetrainModel)

        # Create file menu and add actions
        operation_menu = menu_bar.addMenu("Operation")
        operation_menu.addAction(self.segmentation_act)
        operation_menu.addSeparator()
        operation_menu.addAction(self.annotation_act)
        operation_menu.addSeparator()
        operation_menu.addAction(self.revise_act)
        operation_menu.addSeparator()
        operation_menu.addAction(self.clear_act)
        operation_menu.addSeparator()
        operation_menu.addAction(self.retrain_act)
        # =================================================

        self.setStatusBar(QStatusBar(self))


    def openImageFile(self):
        status_text = self.status_show.text()
        self.status_show.setText("Opening image...")

        """Open an image file and display the contents in the original label widget."""
        self.image_file, _ = QFileDialog.getOpenFileName(self, "Open Image", 
            "", "Images (*.png *.jpeg *.jpg *.bmp)")
        
        if self.image_file:
            self.image = QImage(self.image_file) # Create QImage instance
            # Set the pixmap for the original_layer using the QImage instance
            self.original_layer.back_pixmap = QPixmap(self.image).scaled(
                    self.original_layer.width(), self.original_layer.height(), Qt.KeepAspectRatio)
            # self.original_layer.back_pixmap = QPixmap(self.image).scaled(
            #         self.original_layer.width(), self.original_layer.height(), Qt.KeepAspectRatioByExpanding)
            self.original_layer.updatePixmap()

            self.ruler_layer.ruler_pixmap = self.ruler_layer.ruler_pixmap.scaled(self.original_layer.back_pixmap.size(), Qt.IgnoreAspectRatio)
            self.ruler_layer.updatePixmap()

            # self.adjustSize() # Adjust the size of the main window to better fit its contents   
        else:
            QMessageBox.information(self, "Error",
                "No image opened!", QMessageBox.Ok)
        
        self.status_show.setText(status_text)
            
    def openMaskFile(self):
        status_text = self.status_show.text()
        self.status_show.setText("Opening mask...")

        if not self.image_file:
            QMessageBox.information(self, "Error",
                "Open original image first!", QMessageBox.Ok)
            return
        
        """Open a mask file and display the contents in the annotation label widget."""
        self.mask_file, _ = QFileDialog.getOpenFileName(self, "Open Mask", 
            "", "Images (*.png)")

        if self.mask_file:
            
            load_confirmed = False
            if not os.path.splitext(self.mask_file)[0] == os.path.splitext(self.image_file)[0]:
                load_confirmed = QMessageBox.question(self, "Open Mask?", f"Mask “{self.mask_file}” may not match with “{self.image_file}”, still open?",)
            else:
                load_confirmed = True
            
            if load_confirmed:
                self.setCursor(QCursor(Qt.BusyCursor))

                # set the background and canvas size to the orginal image size
                self.back_layer.back_pixmap.scaled(self.original_layer.back_pixmap.size())
                self.back_layer.updatePixmap()
                self.canvas.mask_pixmap.scaled(self.original_layer.back_pixmap.size())
                self.canvas.updatePixmap()

                self.mask_image = QImage(self.mask_file) # Create QImage instance
                # set the pixel transparency
                w = self.mask_image.width()
                h = self.mask_image.height()
                for i in range(w):
                    for j in range(h):
                        pixel_color = self.mask_image.pixelColor(i, j)
                        if pixel_color == QColor(0, 0, 0):
                            pixel_color.setAlphaF(0)
                            self.mask_image.setPixelColor(i, j, pixel_color)
                        elif pixel_color == QColor(255, 255, 255):
                            pixel_color.setAlphaF(0.6)
                            self.mask_image.setPixelColor(i, j, pixel_color)
                
                # # Set the pixmap for the annotation_label using the QImage instance
                self.canvas.mask_pixmap = QPixmap(self.mask_image).scaled(
                        self.canvas.width(), self.canvas.height(), Qt.KeepAspectRatio)
                self.canvas.updatePixmap() # Qpainter is related to self.mask_pixmap

                # self.adjustSize() # Adjust the size of the main window to better fit its contents
                
                self.setCursor(QCursor(Qt.ArrowCursor))
        else:
            QMessageBox.information(self, "Error",
                "No mask opened!", QMessageBox.Ok)
        
        self.status_show.setText(status_text)
    
    def saveImageMaskFile(self):
        status_text = self.status_show.text()
        self.status_show.setText("Saving image and mask...")

        self.image_mask_save, _ = QFileDialog.getSaveFileName(self, "Save Image & Mask", 
            "", "Images (*.jpeg *.png)")
        
        self.image_save = os.path.splitext(self.image_mask_save)[0] + ".jpeg"
        self.mask_save = os.path.splitext(self.image_mask_save)[0] + ".png"
        
        if self.image_save:
            if os.path.exists(self.image_save): # ask before overwriting
                write_confirmed = QMessageBox.question(self, "Overwrite?", f"File “{self.image_save}” has already existed，overwrite it?",)
            else:
                write_confirmed = True
        
            if write_confirmed:
                image = self.back_layer.back_pixmap.toImage()
                image.save(self.image_save, quality=100)
        
        else:
            QMessageBox.information(self, "Error",
                "No path selected!", QMessageBox.Ok)
        
        if self.mask_save:
            if os.path.exists(self.mask_save): # ask before overwriting
                write_confirmed = QMessageBox.question(self, "Overwrite?", f"File “{self.mask_save}” has already existed，overwrite it?",)
            else:
                write_confirmed = True
        
            if write_confirmed:
                mask = self.canvas.mask_pixmap.toImage()
                w = mask.width()
                h = mask.height()
                for i in range(w):
                    for j in range(h):
                        pixel_color = mask.pixelColor(i, j)
                        pixel_color.setAlphaF(1)
                        mask.setPixelColor(i, j, pixel_color)
                
                mask.save(self.mask_save, quality=100)
        
        else:
            QMessageBox.information(self, "Error",
                "No path selected!", QMessageBox.Ok)
        
        self.status_show.setText(status_text)

    def saveImageFile(self):
        status_text = self.status_show.text()
        self.status_show.setText("Saving image...")

        self.image_save, _ = QFileDialog.getSaveFileName(self, "Save Image", 
            "", "Images (*.jpeg)")
        
        if self.image_save:
            if os.path.exists(self.image_save): # ask before overwriting
                write_confirmed = QMessageBox.question(self, "Overwrite?", f"File “{self.image_save}” has already existed，overwrite it?",)
            else:
                write_confirmed = True
        
            if write_confirmed:
                image = self.back_layer.back_pixmap.toImage()
                image.save(self.image_save, quality=100)
        
        else:
            QMessageBox.information(self, "Error",
                "No path selected!", QMessageBox.Ok)
        
        self.status_show.setText(status_text)

    def saveMaskFile(self):
        status_text = self.status_show.text()
        self.status_show.setText("Saving mask...")

        self.mask_save, _ = QFileDialog.getSaveFileName(self, "Save Mask", 
            "", "Images (*.png)")
        
        if self.mask_save:
            if os.path.exists(self.mask_save): # ask before overwriting
                write_confirmed = QMessageBox.question(self, "Overwrite?", f"File “{self.mask_save}” has already existed，overwrite it?",)
            else:
                write_confirmed = True
        
            if write_confirmed:
                mask = self.canvas.mask_pixmap.toImage()
                w = mask.width()
                h = mask.height()
                for i in range(w):
                    for j in range(h):
                        pixel_color = mask.pixelColor(i, j)
                        pixel_color.setAlphaF(1)
                        mask.setPixelColor(i, j, pixel_color)

                mask.save(self.mask_save, quality=100)

        else:
            QMessageBox.information(self, "错误",
                "No path selected!", QMessageBox.Ok)
        
        self.status_show.setText(status_text)

    def doSegmentation(self):
        if self.image_file:
            status_text = self.status_show.text()
            self.status_show.setText("Doing segmentation...")
            before_cursor = self.cursor()
            self.setCursor(QCursor(Qt.BusyCursor))

            # first transfer QPixmap to QImage, then to PIL image, and finally to tensor
            input = self.original_layer.back_pixmap.toImage()
            input = ImageQt.fromqimage(input)
            input = input.convert('L')
            input_size = input.size
            print(input_size)
            resize = torchvision.transforms.Resize((320, 480))
            input = resize(input)
            tt = torchvision.transforms.ToTensor()
            input = tt(input)
            input = torch.unsqueeze(input, 0)
            input = input.cuda()

            # predict
            self.detect_model.eval()
            output = self.detect_model(input)

            # first transfer tensor to PIL, and then to QIamge & QPixmap
            output = torch.squeeze(output, 0)
            # print(output.size())
            output_b = (output > 0.5) * 1.0
            tp = torchvision.transforms.ToPILImage()
            output_b = tp(output_b)
            
            size_back = torchvision.transforms.Resize((input_size[1], input_size[0]), torchvision.transforms.InterpolationMode.NEAREST)
            output_b = size_back(output_b)
            # output_b = output_b.convert('RGB')
            predict_mask = output_b.toqpixmap()

            predict_mask_image = predict_mask.toImage() # Create QImage instance
            trans_mask_image = QImage(predict_mask_image.size(), QImage.Format_ARGB32)
            # set the pixel transparency
            w = predict_mask_image.width()
            h = predict_mask_image.height()
            for i in range(w):
                for j in range(h):
                    pixel_color = predict_mask_image.pixelColor(i, j)
                    if pixel_color == QColor(0, 0, 0):
                        pixel_color.setAlphaF(0)
                        # print(pixel_color.alphaF())
                        trans_mask_image.setPixelColor(i, j, pixel_color)
                        # print(predict_mask_image.pixelColor(i, j).alphaF())
                    elif pixel_color == QColor(255, 255, 255):
                        pixel_color.setAlphaF(0.6)
                        trans_mask_image.setPixelColor(i, j, pixel_color)

            self.back_layer.back_pixmap = self.original_layer.back_pixmap
            self.back_layer.updatePixmap()
            self.canvas.mask_pixmap = QPixmap(trans_mask_image)
            self.canvas.updatePixmap()

            self.status_show.setText(status_text)
            self.setCursor(before_cursor)
            
        else:
            QMessageBox.information(self, "Error",
                "No image opened!", QMessageBox.Ok)
            
        

    def doAnnotation(self):
        # first copy the original image as background
        if self.image_file:
            # Set the pixmap for the annotation_label using the QImage instance
            self.back_layer.back_pixmap = self.original_layer.back_pixmap
            self.back_layer.updatePixmap()
            # print(self.back_layer.back_pixmap.size())
            # print(self.back_layer.size())
            self.canvas.mask_pixmap = self.canvas.mask_pixmap.scaled(self.back_layer.back_pixmap.size(), Qt.IgnoreAspectRatio)
            self.canvas.updatePixmap()
            # print(self.canvas.mask_pixmap.size())

            # self.adjustSize() # Adjust the size of the main window to better fit its contents   

            self.canvas.is_annotation = True
            self.is_revise_button.setChecked(True)
            self.canvas.setCursor(QCursor(Qt.CrossCursor)) # become cross for painting
            self.pen_button.setEnabled(self.canvas.is_annotation)
            self.eraser_button.setEnabled(self.canvas.is_annotation)
            self.pen_width_box.setEnabled(self.canvas.is_annotation)
            # self.eraser_width_box.setEnabled(self.canvas.is_annotation)
            self.pen_button.setChecked(True) # default painter is pen

            self.status_show.setText("Doing annotation...")
        else:
            QMessageBox.information(self, "Error",
                "No image opened!", QMessageBox.Ok)


    def doRevise(self):
        if self.mask_file:
            self.canvas.is_annotation = True
            self.is_revise_button.setChecked(True)
            self.canvas.setCursor(QCursor(Qt.CrossCursor)) # become cross for painting
            self.pen_button.setEnabled(self.canvas.is_annotation)
            self.eraser_button.setEnabled(self.canvas.is_annotation)
            self.pen_width_box.setEnabled(self.canvas.is_annotation)
            # self.eraser_width_box.setEnabled(self.canvas.is_annotation)
            self.pen_button.setChecked(True) # default painter is pen

            self.status_show.setText("Do rivision...")
        else:
            QMessageBox.information(self, "Error",
                "No mask can be revised!", QMessageBox.Ok)

    def doClear(self):
        self.original_layer.back_pixmap = self.default_back_pixmap
        self.original_layer.updatePixmap()
        self.ruler_layer.ruler_pixmap = QPixmap(QSize(800, 600))
        self.ruler_layer.ruler_pixmap.fill(self.ruler_layer.back_color)
        self.ruler_layer.updatePixmap()
        self.ruler_layer.pix_per_length = 0
        self.back_layer.back_pixmap = self.default_back_pixmap
        self.back_layer.updatePixmap()
        self.canvas.mask_pixmap = QPixmap(QSize(800, 600))
        self.canvas.mask_pixmap.fill(self.canvas.back_color)
        self.canvas.updatePixmap()

        self.image_file = None
        self.image = None
        self.mask_file = None
        self.mask_image = None

        self.is_revise_button.setChecked(False)
        self.pen_button.setEnabled(False)
        self.eraser_button.setEnabled(False)
        self.pen_width_box.setEnabled(False)
        self.eraser_width_box.setEnabled(False)
        
        self.status_show.setText("")
    
    def doRetrainModel(self):
        status_text = self.status_show.text()
        self.status_show.setText("Updating model parameters...")
        before_cursor = self.cursor()
        self.setCursor(Qt.BusyCursor)

        # some basic settings
        train_on_gpu = torch.cuda.is_available()
        batch_size = 16
        num_workers = 0
        n_epochs = 10
        optimizer_type = 'Adam' """Either Adam or SGD, adjust the learning rate in the
                                "Specify the loss function and optimizer" section"""
        criterion_type = 'TverskyLoss'  """ Adjust the penalties in the "Specify the loss 
                                        function and optimizer" section"""
        
        # make dataset
        image_path = {}
        for c in ['set1', 'set2']:
            image_path[c] = glob('dataset/' + c + '/*.jpeg',recursive=True)

        fucai_defect_dataset ={}
        fucai_defect_dataset['train'] = DefectDetectionDataset (image_path['set1'], 'train')
        fucai_defect_dataset['val'] = DefectDetectionDataset (image_path['set2'], 'val')
        
        loaders={}
        loaders['train'] = torch.utils.data.DataLoader(fucai_defect_dataset['train'], 
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)
        loaders['val'] = torch.utils.data.DataLoader(fucai_defect_dataset['val'], 
                                                    batch_size=batch_size,
                                                    shuffle=False, 
                                                    num_workers=num_workers)
        
        # set train details
        self.detect_model.load_state_dict(torch.load('model.pt'))
        if train_on_gpu:
            self.detect_model.cuda()

        positive_weight = 0
        negative_weight = 0
        total_pixels = 0
        for _, target in fucai_defect_dataset['train']:
            positive_weight += ((target.cpu().numpy()) >= 0.5).sum()
            negative_weight += ((target.cpu().numpy()) < 0.5).sum()
            total_pixels += (320 * 480)
        positive_weight /= total_pixels
        negative_weight /= total_pixels
        print('positive weight = ',positive_weight, '\tnegative weight = ', negative_weight)

        if criterion_type == 'WeightedBCE':
            weight = np.array([negative_weight, positive_weight])
            weight = torch.from_numpy(weight)
            criterion = WeightedBCELoss(weights=weight)
        else:
            criterion = TverskyLoss(1e-10,0.3,.7)
        if optimizer_type == 'SGD':
            optimizer = optim.SGD(self.detect_model.parameters(), lr=0.00005, momentum=0.9)
        else:
            optimizer = optim.Adam(self.detect_model.parameters(), lr = 0.0001)
        
        # train the model
        # WARN!!!!!!
        # if you are very sure to overwrite the original parameter, change the parameter file to "model.pt"
        self.detect_model = train_2D(n_epochs, loaders, self.detect_model, optimizer, criterion, train_on_gpu, 'model_retrain.pt')
        loss=pd.read_csv('loss_epoch.csv',header=0,index_col=False)
        plt.plot(loss['epoch'],loss['Training Loss'],'r',loss['epoch'],loss['Validation Loss'],'g')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend(labels=['Train','Valid'])
        plt.show()
        plt.savefig('loss_epoch.png')

        self.detect_model.load_state_dict(torch.load('model_retrain.pt'))

        self.status_show.setText(status_text)
        self.setCursor(before_cursor)
        

    def setupToolbar(self):
        toolbar = QToolBar("Operation")
        toolbar.setContextMenuPolicy(Qt.PreventContextMenu) # cannot right click to hide, see https://realpython.com/python-menus-toolbars/#building-context-or-pop-up-menus-in-pyqt
        toolbar.setMovable(False) # cannot move
        toolbar.setFloatable(False) # cannot float
        self.addToolBar(toolbar)
        
        # actions have been defined in setupMenu
        toolbar.addSeparator()
        toolbar.addAction(self.segmentation_act)
        toolbar.addSeparator()
        toolbar.addAction(self.annotation_act)
        toolbar.addSeparator()
        toolbar.addAction(self.revise_act)
        toolbar.addSeparator()
        toolbar.addAction(self.clear_act)
        toolbar.addSeparator()
        toolbar.addAction(self.retrain_act)
        toolbar.addSeparator()


class Background(QLabel):
    def __init__(self, img_size):
        super().__init__()
        self.back_pixmap = QPixmap(img_size)
        self.back_pixmap.fill(QColor(48, 76, 98))
        self.updatePixmap()

        # for transformation
        # =========================================
        # QPainter coord
        self.lefttop = QPointF(0, 0)
        self.moved_lefttop = QPointF(0, 0)
        # QWidget coord
        self.original_center = QPointF(img_size.width() / 2, img_size.height() / 2)
        self.start_pos = None
        self.end_pos = None
        self.middle_click = False
        self.scale = 1

        # reference of partner canvas and background
        # all the images transform together
        self.canvas_partner = None
        self.background_partner = None
        # =========================================

    
    def updatePixmap(self):
        self.setPixmap(self.back_pixmap)
        self.update()

    # see https://blog.csdn.net/hi_sir_destroy/article/details/120049703
    # ====================================================================
    def mousePressEvent(self, e):
        """
        mouse press events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.MiddleButton:
            self.middle_click = True
            self.start_pos = e.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
 
    def mouseReleaseEvent(self, e):
        """
        mouse release events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.MiddleButton:
            self.middle_click = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def wheelEvent(self, e):
        angle = e.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        # 获取当前鼠标相对于view的位置
        if angleY > 0:
            self.scale *= 1.1
        else:  # 滚轮下滚
            self.scale *= 1 / 1.1
        # self.adjustSize()
        self.update()
    
    def mouseMoveEvent(self, e):
        """
        mouse move events for the widget
        :param e: QMouseEvent
        :return:
        """
        if self.middle_click:
            self.end_pos = e.pos()
            # QPainter coord
            move_distance = (self.end_pos - self.start_pos) / self.scale # scale the distance
            self.lefttop = self.lefttop + move_distance
            self.start_pos = e.pos()
            self.update()
    
    def paintEvent(self, e):
        """
        receive paint events
        :param e: QPaintEvent
        :return:
        """
        if self.back_pixmap:
            scale_painter = QPainter()
            scale_painter.begin(self)
            scale_painter.scale(self.scale, self.scale)
            # QPainter coord
            # move to the canter of widget to display
            move_x = self.original_center.x() / self.scale - self.back_pixmap.width() / 2
            move_y = self.original_center.y() / self.scale - self.back_pixmap.height() / 2
            self.moved_lefttop = QPointF(self.lefttop.x() + move_x, self.lefttop.y() + move_y)
            # QPainter coord
            scale_painter.drawPixmap(self.moved_lefttop, self.back_pixmap)
            scale_painter.end()
            self.updateCanvasPartner()
            self.updateBackgroundPartner()
    # ====================================================================

    def updateCanvasPartner(self):
        if self.canvas_partner:
            self.canvas_partner.lefttop = self.lefttop
            self.canvas_partner.start_pos = self.start_pos
            self.canvas_partner.end_pos = self.end_pos
            self.canvas_partner.scale = self.scale

            self.canvas_partner.update()
    
    def updateBackgroundPartner(self):
        if self.background_partner:
            self.background_partner.lefttop = self.lefttop
            self.background_partner.start_pos = self.start_pos
            self.background_partner.end_pos = self.end_pos
            self.background_partner.scale = self.scale

            self.background_partner.update()


class Ruler(QLabel):
    def __init__(self, img_size):
        super().__init__()
        self.ruler_pixmap = QPixmap(img_size)
        self.back_color = QColor(0, 0, 0)
        self.back_color.setAlphaF(0)
        self.ruler_pixmap.fill(self.back_color)
        self.updatePixmap()

        self.first_x, self.first_y, self.second_x, self.second_y = None, None, None, None
        self.is_first_point = True
        self.line_color = QColor("#000080")
        self.point_color = QColor("#0000ff")

        self.line_width = 2
        self.point_width = 4

        self.is_annotation = False
        self.is_ruler = False
        self.is_measure = False

        # for transformation
        # =========================================
        # QPainter coord
        self.lefttop = QPointF(0, 0)
        self.moved_lefttop = QPointF(0, 0)
        # QWidget coord
        self.original_center = QPointF(img_size.width() / 2, img_size.height() / 2)
        self.start_pos = None
        self.end_pos = None
        self.middle_click = False
        self.scale = 1

        # reference of partner canvas and background
        # all the images transform together
        self.background_partner = None
        # =========================================

        self.entered_length = 0
        self.selected_pix = 0
        self.pix_per_length = 0
        self.related_ruler_label = QLabel()
        self.related_measure_label = QLabel()

    
    def updatePixmap(self):
        self.setPixmap(self.ruler_pixmap)
        self.update()

    # see https://blog.csdn.net/hi_sir_destroy/article/details/120049703
    # ====================================================================
    def mousePressEvent(self, e):
        """
        mouse press events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.MiddleButton:
            self.middle_click = True
            self.start_pos = e.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        
        if e.button() == Qt.LeftButton:
            if self.is_annotation:
                if self.is_measure:
                    if self.pix_per_length == 0:
                        QMessageBox.information(self, "Error",
                            "No scale added!", QMessageBox.Ok)
                        self.is_measure = False
                        self.setCursor(QCursor(Qt.ArrowCursor))
                        self.is_annotation = False
                        return

                if self.is_first_point:
                    self.ruler_pixmap.fill(self.back_color)
                    self.updatePixmap()
                    self.first_x = (e.x() - self.moved_lefttop.x() * self.scale) / self.scale
                    self.first_y = (e.y() - self.moved_lefttop.y() * self.scale) / self.scale
                    self.point_painter = QPainter(self.ruler_pixmap)
                    self.setPixmap(self.ruler_pixmap)
                    self.pp = self.point_painter.pen()
                    self.pp.setWidth(self.point_width)
                    self.pp.setColor(self.point_color)
                    self.point_painter.setPen(self.pp)
                    self.point_painter.drawPoint(QPointF(self.first_x, self.first_y))
                    self.point_painter.end()
                    self.update()
                    self.is_first_point = False
                else:
                    self.second_x = (e.x() - self.moved_lefttop.x() * self.scale) / self.scale
                    self.second_y = (e.y() - self.moved_lefttop.y() * self.scale) / self.scale
                    self.point_painter = QPainter(self.ruler_pixmap)
                    self.setPixmap(self.ruler_pixmap)
                    self.pp = self.point_painter.pen()
                    self.pp.setWidth(self.point_width)
                    self.pp.setColor(self.point_color)
                    self.point_painter.setPen(self.pp)
                    self.point_painter.drawPoint(QPointF(self.second_x, self.second_y))
                    self.point_painter.end()
                    self.update()

                    self.line_painter = QPainter(self.ruler_pixmap)
                    self.setPixmap(self.ruler_pixmap)
                    self.lp = self.line_painter.pen()
                    self.lp.setWidth(self.line_width)
                    self.lp.setColor(self.line_color)
                    self.line_painter.setPen(self.lp)
                    self.line_painter.drawLine(QPointF(self.first_x, self.first_y), QPointF(self.second_x, self.second_y))
                    self.line_painter.end()
                    self.update()

                    self.selected_pix = math.sqrt((self.first_x - self.second_x) ** 2 + (self.first_y - self.second_y) ** 2)

                    if self.is_ruler:
                        self.entered_length, ok = QInputDialog.getDouble(self, "Pick Length", "Input the length of the the selected line (mm): ", value=0, min=0, decimals=2,)
                        if ok:
                            try:
                                self.pix_per_length = self.selected_pix / self.entered_length
                                self.related_ruler_label.setText(f"Scale is {self.pix_per_length:.2f} pix/mm")
                            except ZeroDivisionError:
                                QMessageBox.information(self, "Error",
                                "Length not valid!", QMessageBox.Ok)
                        else:
                            QMessageBox.information(self, "Error",
                                "Length not valid!", QMessageBox.Ok)
                        self.is_ruler = False

                    if self.is_measure:
                        length = self.selected_pix / self.pix_per_length
                        self.related_measure_label.setText(f"Length is {length:.2f} mm")
                        self.is_measure = False

                    self.setCursor(QCursor(Qt.ArrowCursor))
                    self.is_first_point = True
                    self.is_annotation = False
 
 
    def mouseReleaseEvent(self, e):
        """
        mouse release events for the widget
        :param e: QMouseEvent
        :return:
        """
        if e.button() == Qt.MiddleButton:
            self.middle_click = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def wheelEvent(self, e):
        angle = e.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        # 获取当前鼠标相对于view的位置
        if angleY > 0:
            self.scale *= 1.1
        else:  # 滚轮下滚
            self.scale *= 1 / 1.1
        # self.adjustSize()
        self.update()
    
    def mouseMoveEvent(self, e):
        """
        mouse move events for the widget
        :param e: QMouseEvent
        :return:
        """
        if self.middle_click:
            self.end_pos = e.pos()
            # QPainter coord
            move_distance = (self.end_pos - self.start_pos) / self.scale # scale the distance
            self.lefttop = self.lefttop + move_distance
            self.start_pos = e.pos()
            self.update()
    
    def paintEvent(self, e):
        """
        receive paint events
        :param e: QPaintEvent
        :return:
        """
        if self.ruler_pixmap:
            scale_painter = QPainter()
            scale_painter.begin(self)
            scale_painter.scale(self.scale, self.scale)
            # QPainter coord
            # move to the canter of widget to display
            move_x = self.original_center.x() / self.scale - self.ruler_pixmap.width() / 2
            move_y = self.original_center.y() / self.scale - self.ruler_pixmap.height() / 2
            self.moved_lefttop = QPointF(self.lefttop.x() + move_x, self.lefttop.y() + move_y)
            # QPainter coord
            scale_painter.drawPixmap(self.moved_lefttop, self.ruler_pixmap)
            scale_painter.end()
            self.updateBackgroundPartner()
    # ====================================================================
    
    def updateBackgroundPartner(self):
        if self.background_partner:
            self.background_partner.lefttop = self.lefttop
            self.background_partner.start_pos = self.start_pos
            self.background_partner.end_pos = self.end_pos
            self.background_partner.scale = self.scale

            self.background_partner.update()


class Canvas(QLabel):
    def __init__(self, img_size):
        super().__init__()
        self.mask_pixmap = QPixmap(img_size)
        self.back_color = QColor(0, 0, 0)
        self.back_color.setAlphaF(0)
        self.mask_pixmap.fill(self.back_color)
        self.updatePixmap()

        self.last_x, self.last_y = None, None
        self.pen_color = QColor("#ffffff")
        self.pen_color.setAlphaF(0.6)

        self.pen_width = 5

        self.is_annotation = False
        self.is_eraser = False

        self.left_click = False

        # for transformation
        # =========================================
        # QPainter coord
        self.lefttop = QPointF(0, 0)
        self.moved_lefttop = QPointF(0, 0)
        # QWidget coord
        self.original_center = QPointF(img_size.width() / 2, img_size.height() / 2)
        self.start_pos = None
        self.end_pos = None
        self.middle_click = False
        self.scale = 1

        # reference of partner background
        # all the images transform together
        self.background_partner = None
        # =========================================

    def updatePixmap(self):
        self.setPixmap(self.mask_pixmap)
        self.update()

    def set_pen_color(self, c):
        self.pen_color = QColor(c)
        self.pen_color.setAlphaF(0.6)

    def mouseMoveEvent(self, e):
        if self.left_click:
            # if self.last_x is None:  # First event.
            #     return  # Ignore the first time.

            # print(e.x(), e.y())
            # print(self.lefttop.x(), self.lefttop.y())

            if not self.is_annotation: # the painter only work when annotation
                return

            # QWidget coord
            now_x = (e.x() - self.moved_lefttop.x() * self.scale) / self.scale
            now_y = (e.y() - self.moved_lefttop.y() * self.scale) / self.scale

            # painter = QPainter(self.pixmap())
            self.painter = QPainter(self.mask_pixmap)
            self.setPixmap(self.mask_pixmap)
            self.p = self.painter.pen()
            self.p.setWidth(self.pen_width)
        
            if self.is_eraser:
                # refer to https://blog.csdn.net/weixin_47878978/article/details/113174513
                self.painter.setCompositionMode(QPainter.CompositionMode_Clear)
                self.p.setColor(self.back_color)
            else:
                self.painter.setCompositionMode(QPainter.CompositionMode_Source)
                self.p.setColor(self.pen_color)
            self.painter.setPen(self.p)
            self.painter.drawLine(QPointF(self.last_x, self.last_y), QPointF(now_x, now_y))
            self.painter.end()
            self.update()

            # Update the origin for next time.
            self.last_x = now_x
            self.last_y = now_y
        
        elif self.middle_click:
            self.end_pos = e.pos()
            # QPainter coord
            move_distance = (self.end_pos - self.start_pos) / self.scale # scale the distance
            self.lefttop = self.lefttop + move_distance
            self.start_pos = e.pos()
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.last_x = (e.x() - self.moved_lefttop.x() * self.scale) / self.scale
            self.last_y = (e.y() - self.moved_lefttop.y() * self.scale) / self.scale
            self.painter = QPainter(self.mask_pixmap)
            self.setPixmap(self.mask_pixmap)
            self.p = self.painter.pen()
            self.p.setWidth(self.pen_width)
        
            if self.is_eraser:
                # refer to https://blog.csdn.net/weixin_47878978/article/details/113174513
                self.painter.setCompositionMode(QPainter.CompositionMode_Clear)
                self.p.setColor(self.back_color)
            else:
                self.painter.setCompositionMode(QPainter.CompositionMode_Source)
                self.p.setColor(self.pen_color)
            self.painter.setPen(self.p)
            if self.is_annotation:
                self.painter.drawPoint(QPointF(self.last_x, self.last_y))
            self.painter.end()
            self.update()
        elif e.button() == Qt.MiddleButton:
            self.middle_click = True
            self.start_pos = e.pos()
            self.cursor_before_move = self.cursor()
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False
            self.last_x = None
            self.last_y = None
        if e.button() == Qt.MiddleButton:
            self.middle_click = False
            self.setCursor(self.cursor_before_move)
    
    def wheelEvent(self, e):
        angle = e.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        # 获取当前鼠标相对于view的位置
        if angleY > 0:
            self.scale *= 1.1
        else:  # 滚轮下滚
            self.scale *= 1 / 1.1
        # self.adjustSize()
        self.update()

        # print(self.mask_pixmap.size())
    
    def paintEvent(self, e):
        if self.mask_pixmap:
            scale_painter = QPainter()
            scale_painter.begin(self)
            scale_painter.scale(self.scale, self.scale)
            # QPainter coord
            # move to the canter of widget to display
            move_x = self.original_center.x() / self.scale - self.mask_pixmap.width() / 2
            move_y = self.original_center.y() / self.scale - self.mask_pixmap.height() / 2
            self.moved_lefttop = QPointF(self.lefttop.x() + move_x, self.lefttop.y() + move_y)
            # QPainter coord
            scale_painter.drawPixmap(self.moved_lefttop, self.mask_pixmap)
            scale_painter.end()
            self.updateBackgroundPartner()
    
    def updateBackgroundPartner(self):
        if self.background_partner:
            self.background_partner.lefttop = self.lefttop
            self.background_partner.start_pos = self.start_pos
            self.background_partner.end_pos = self.end_pos
            self.background_partner.scale = self.scale

            self.background_partner.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    extra = {
                # Font
                'font_family': 'Microsoft Yahei',
                'font_size': 16,
            }
    apply_stylesheet(app, theme='dark_teal.xml', extra=extra) # https://zhuanlan.zhihu.com/p/390192953
    window = MainWindow()
    window.show()
    app.exec_()


