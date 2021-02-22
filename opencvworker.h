/*****************************************************************************************
 * Copyright (C) 2021 Renesas Electronics Corp.
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

#ifndef OPENCVCAPTUREWORKER_H
#define OPENCVCAPTUREWORKER_H

#include <opencv2/videoio.hpp>
#include <QObject>
#include <memory>

#define CAMERA_WIDTH "800"
#define CAMERA_HEIGHT "600"

class opencvWorker : public QObject
{
    Q_OBJECT

signals:
    void sendImage(const cv::Mat&);
    void webcamInit(bool webcamInitialised);

private:
    std::unique_ptr<cv::VideoCapture> videoCapture;
    bool webcamInitialised;

private slots:
    void initialiseWebcam(QString cameraLocation);
    void readFrame();
    void disconnectWebcam();
};

#endif // OPENCVCAPTUREWORKER_H
