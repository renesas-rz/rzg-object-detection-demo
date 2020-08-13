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

#include "opencvworker.h"

void opencvWorker::initialiseWebcam(QString cameraLocation)
{
    QString gstreamerPipeline;

    if (webcamInitialised == true)
        videoCapture->release();

    if (!cameraLocation.isEmpty()) {
        gstreamerPipeline = "v4l2src device=" + QString(cameraLocation) + \
            " ! video/x-raw, width=" CAMERA_WIDTH ", height=" CAMERA_HEIGHT \
            " ! videoconvert ! appsink";
        videoCapture.reset(new cv::VideoCapture(gstreamerPipeline.toStdString(), \
                                                cv::CAP_GSTREAMER));
        if (videoCapture->isOpened()) {
            webcamInitialised = true;
        } else {
            webcamInitialised = false;
        }
    } else {
        webcamInitialised = false;
    }

    emit webcamInit(webcamInitialised);
}

void opencvWorker::readFrame()
{
    cv::Mat videoFrame;
    if(!videoCapture->read(videoFrame)) {
        webcamInitialised = false;
        videoCapture->release();
        emit webcamInit(webcamInitialised);
        return;
    }

    emit sendImage(videoFrame);
}

void opencvWorker::disconnectWebcam()
{
    if (webcamInitialised == true)
        videoCapture->release();
}
