/*****************************************************************************************
 * Copyright (C) 2020 Renesas Electronics Corp.
 * This file is part of the RZG Object Detection Demo.
 *
 * The RZG Object Detection Demo is free software using the Qt Open Source Model: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * The RZG Object Detection Demo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the RZG Object Detection Demo.  If not, see <https://www.gnu.org/licenses/>.
 *****************************************************************************************/

#include <QApplication>
#include <QCommandLineParser>
#include <QFile>
#include <QDir>

#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QCommandLineParser parser;
    QCommandLineOption cameraOption(QStringList() << "c" << "camera","Choose a camera.","file");
    QCommandLineOption labelOption(QStringList() << "l" << "label","Choose a label file.","file");
    QCommandLineOption modelOption(QStringList() << "m" << "model","Choose a model.","file");
    QCommandLineOption tpuOption(QStringList() << "t" << "tpu","Enable tpu processing.");
    QString cameraLocation;
    QString labelLocation;
    QString modelLocation;
    bool tpu;
    QDir dir;
    QStringList filesAtPwd;
    QString labelFileName;
    QString applicationDescription = \
    "Object Detection Demo\n"\
    "  Draws boxes around detected objects and displays the name and\n"\
    "  confidence of the object. Also displays inference time and FPS\n"\
    "  if applicable.\n\n"\
    "Required Hardware:\n"\
    "  Camera: Currently supports Logitech C922 Pro Stream, should\n"\
    "          work with any UVC compatible USB camera that has a\n"\
    "          supported resolution of 800x600.\n"\
    "  TPU Mode: Requires Coral USB Accelerator.\n\n"\
    "Supported Models:\n"\
    "  MobileNet v2 SSD Quantised TensorFlow Lite\n\n"\
    "Buttons:\n"\
    "  Run: Run inference on the selected image once.\n"\
    "  Load File: Load an image or video from the filesystem. Media formats\n"\
    "             supported by OpenCV 4.1.1 and GStreamer 1.12.2 can be\n"\
    "             opened, including mp4, m4v, mkv, webm, bmp, jpg, png.\n"\
    "  Video Controls: Only available when video is loaded. Play,\n"\
    "                  pause, stop, and seek can be performed.\n"\
    "  Load Camera: Load a camera stream.\n"\
    "  Capture Image: Capture an image from the camera.\n"\
    "  Continuous Checkbox: Only available when a camera stream or video\n"\
    "                       is loaded. Enable to continuously run inference.\n"\
    "  Stop: Stop continuous inference.\n"\
    "  Threads: Only available in CPU mode. Change the number of inference\n"\
    "           threads.\n"\
    "  About->License: Read the license that this app is licensed under.\n"\
    "  About->Exit: Close the application.\n"\
    "  Camera->Reset: Reset the connection to the camera.\n"\
    "  Camera->Disconnect: Disconnect the currently connected camera.\n\n"\
    "Default options:\n"\
    "  Camera: /dev/v4l/by-id/<first file>\n"\
    "  Label: ./*label*.txt\n"\
    "  Model:\n"\
    "    CPU Mode: ./mobilenet_ssd_v2_coco_quant_postprocess.tflite\n"\
    "    TPU Mode: ./mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite\n"\
    "  TPU Mode: Disabled";
    parser.addOption(cameraOption);
    parser.addOption(labelOption);
    parser.addOption(modelOption);
    parser.addOption(tpuOption);
    parser.addHelpOption();
    parser.setApplicationDescription(applicationDescription);
    parser.process(a);
    cameraLocation = parser.value(cameraOption);
    labelLocation = parser.value(labelOption);
    modelLocation = parser.value(modelOption);
    tpu = parser.isSet(tpuOption);

    if (cameraLocation.isEmpty() && QDir("/dev/v4l/by-id").exists()) {
        cameraLocation = QDir("/dev/v4l/by-id").entryInfoList(\
                    QDir::NoDotAndDotDot).at(0).absoluteFilePath();
    }

    if (labelLocation.isEmpty()) {
        dir.setPath(QDir::currentPath());
        filesAtPwd = dir.entryList(QStringList("*label*.txt"));
        if (filesAtPwd.isEmpty()) {
            qFatal("Label txt file not found in current directory,  please specify a file with -l");
        }
        labelLocation = filesAtPwd.at(0);
    } else {
        if (!QFile::exists(labelLocation))
            qFatal("%s not found.", labelLocation.toStdString().c_str());
    }

    if (modelLocation.isEmpty()) {
        if (tpu){
            modelLocation = TPU_MODEL_NAME;
            if (!QFile::exists(modelLocation))
                qFatal("%s not found in the current directory, please specify a tflite TPU model with -m", \
                       modelLocation.toStdString().c_str());
        } else {
            modelLocation = CPU_MODEL_NAME;
            if (!QFile::exists(modelLocation))
                qFatal("%s not found in the current directory, please specify a tflite model with -m", \
                       modelLocation.toStdString().c_str());
        }
    }

    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    MainWindow w(nullptr, cameraLocation, labelLocation, modelLocation, tpu);
    w.show();
    return a.exec();
}
