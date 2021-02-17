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

#include <QGraphicsScene>
#include <QGraphicsTextItem>
#include <QFileDialog>
#include <QMessageBox>
#include <QThread>
#include <QEventLoop>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "tfliteworker.h"
#include "opencvworker.h"

MainWindow::MainWindow(QWidget *parent, QString cameraLocation, QString labelLocation, \
                       QString modelLocation, bool tpuEnable)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    QFile labelFile;
    QString fileLine;

    webcamName = cameraLocation;
    ui->setupUi(this);
    QMainWindow::showMaximized();
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    ui->stopButton->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    ui->videoSlider->setTracking(true);

    ui->playButton->setVisible(false);
    ui->stopButton->setVisible(false);
    ui->videoSlider->setVisible(false);

    connect(ui->videoSlider, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));

    labelFile.setFileName(labelLocation);
    if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text))
        qFatal("%s could not be opened.", labelLocation.toStdString().c_str());

    while (!labelFile.atEnd()) {
        fileLine = labelFile.readLine();
        fileLine.remove(QRegularExpression("^\\s*\\d*\\s*"));
        fileLine.remove(QRegularExpression("\n"));
        labelList.append(fileLine);
    }

    ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    ui->checkBoxContinuous->setEnabled(false);
    ui->pushButtonWebcam->setEnabled(false);
    ui->pushButtonCapture->setEnabled(false);

    if (tpuEnable) {
        ui->inferenceThreadCount->setValue(1);
        ui->inferenceThreadCount->setEnabled(false);
        inferenceTimeLabel = "TPU Inference Time: ";
    } else {
        inferenceTimeLabel = "CPU Inference Time: ";
    }
    cameraStatusLabel = "Camera status: ";

    ui->labelInference->setText(inferenceTimeLabel);
    ui->labelCamera->setText(cameraStatusLabel);

    imageLoaded = false;
    videoLoaded = false;

    qRegisterMetaType<cv::Mat>();
    opencvThread = new QThread();
    opencvThread->setObjectName("opencvThread");
    opencvThread->start();
    cvWorker = new opencvWorker();
    cvWorker->moveToThread(opencvThread);
    connect(ui->pushButtonWebcam, SIGNAL(toggled(bool)), this, \
            SLOT(pushButtonWebcamCheck(bool)));
    connect(cvWorker, SIGNAL(sendImage(const cv::Mat&)), this, SLOT(showImage(const cv::Mat&)));
    connect(cvWorker, SIGNAL(webcamInit(bool)), this, SLOT(webcamInitStatus(bool)));

    QMetaObject::invokeMethod(cvWorker, "initialiseWebcam", Qt::AutoConnection, Q_ARG(QString,webcamName));

    qRegisterMetaType<QVector<float> >("QVector<float>");
    tfliteThread = new QThread();
    tfliteThread->setObjectName("tfliteThread");
    tfliteThread->start();
    tfWorker = new tfliteWorker(tpuEnable,modelLocation);
    tfWorker->moveToThread(tfliteThread);
    connect(tfWorker, SIGNAL(requestImage()), this, SLOT(receiveRequest()));
    connect(this, SIGNAL(sendImage(const cv::Mat&)), tfWorker, SLOT(receiveImage(const cv::Mat&)));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>&, int, const cv::Mat&)), \
            this, SLOT(receiveOutputTensor(const QVector<float>&, int, const cv::Mat&)));
    connect(this, SIGNAL(sendNumOfInferenceThreads(int)), tfWorker, SLOT(receiveNumOfInferenceThreads(int)));

    webcamTimer = new QTimer();
    fpsTimer = new QElapsedTimer();
    videoContinuousTimer = new QElapsedTimer();
    videoTimer = new QTimer();
    connect(videoTimer, SIGNAL(timeout()), this, SLOT(getVideoFileFrame()));
}

void MainWindow::on_pushButtonFile_clicked()
{
    qeventLoop = new QEventLoop;
    QString fileName;
    QStringList fileNames;
    QFileDialog dialog(this);
    QString fileFilter;
    QString fileNameFull;

    connect(this, SIGNAL(fileLoaded()), qeventLoop, SLOT(quit()));

    on_stopButton_clicked();
    ui->pushButtonWebcam->setChecked(false);
    outputTensor.clear();
    ui->labelInference->setText(inferenceTimeLabel);
    dialog.setFileMode(QFileDialog::AnyFile);

    fileFilter = "Images (*.bmp *.dib *.jpeg *.jpg *.jpe *.png *.pbm *.pgm *.ppm *.sr *.ras *.tiff *.tif);;";
    fileFilter += "Videos (*.asf *.avi *.3gp *.mp4 *m4v *.mov *.flv *.mpeg *.mkv *.webm *.mxf *.ogg)";

    dialog.setNameFilter(fileFilter);
    dialog.setViewMode(QFileDialog::Detail);

    if (dialog.exec())
        fileNames = dialog.selectedFiles();

    if(fileNames.count() > 0)
        fileName = fileNames.at(0);

    if (QFile::exists(fileName)) {
        fileNameFull = QDir::current().absoluteFilePath(fileName);
        if (dialog.selectedNameFilter().contains("Images")) {
            matToSend = cv::imread(fileNameFull.toStdString());
            drawMatToView(matToSend);

            imageLoaded = true;
            videoLoaded = false;
            ui->checkBoxContinuous->setCheckState(Qt::Unchecked);
            ui->checkBoxContinuous->setEnabled(false);
            ui->playButton->setVisible(false);
            ui->stopButton->setVisible(false);
            ui->videoSlider->setVisible(false);
        }
        else if (dialog.selectedNameFilter().contains("Videos")) {
            cap = cv::VideoCapture(fileNameFull.toStdString());
            videoLoaded = false;
            imageLoaded = false;
            getVideoFileFrame();
            ui->checkBoxContinuous->setEnabled(true);
            ui->playButton->setVisible(true);
            ui->stopButton->setVisible(true);
            ui->videoSlider->setVisible(true);
            ui->videoSlider->setMaximum(cap.get(cv::CAP_PROP_FRAME_COUNT));
            fpsTimer->start();
            videoContinuousTimer->start();
        }
    }
    else {
        QMessageBox::warning(this, "Warning", "File does not exist.");
    }

    emit fileLoaded();
    qeventLoop->exec();
}

void MainWindow::on_pushButtonRun_clicked()
{
    if (!imageLoaded && !videoLoaded) {
            QMessageBox::warning(this, "Warning", "No source selected, please select a file.");
            return;
    }

    if (ui->checkBoxContinuous->isChecked()) {
        continuousMode = true;
        fpsTimer->start();
    }

    if (videoLoaded) {
        videoTimer->stop();
        ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        ui->playButton->setChecked(false);
    }

    QMetaObject::invokeMethod(tfWorker, "process");
    ui->pushButtonRun->setEnabled(false);
}

void MainWindow::on_inferenceThreadCount_valueChanged(int threads)
{
    emit sendNumOfInferenceThreads(threads);
}

void MainWindow::receiveRequest()
{
    sendImage(matToSend);
    tfWorker->moveToThread(tfliteThread);
}

void MainWindow::receiveOutputTensor(const QVector<float>& receivedTensor, int receivedTimeElapsed, const cv::Mat& receivedMat)
{
    if (ui->pushButtonRun->isEnabled())
        return;

    outputTensor = receivedTensor;

    ui->labelInference->setText(inferenceTimeLabel + QString("%1 ms").arg(receivedTimeElapsed));

    if (!ui->pushButtonWebcam->isChecked() && !ui->playButton->isChecked())
       drawMatToView(receivedMat);

    if (!ui->checkBoxContinuous->isChecked()) {
        ui->pushButtonRun->setEnabled(true);
    } else {
        drawMatToView(receivedMat);
        drawFPS(fpsTimer->restart());
        if (videoLoaded) {
            getVideoFileFrame();
            ui->playButton->setChecked(true);
        }

        QMetaObject::invokeMethod(tfWorker, "process");
    }

    drawBoxes();
}

void MainWindow::on_pushButtonStop_clicked()
{
    ui->pushButtonRun->setEnabled(true);
    continuousMode = false;
    outputTensor.clear();
    if(videoLoaded) {
        ui->playButton->setChecked(false);
        ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        videoTimer->stop();
    }
}

void MainWindow::on_pushButtonCapture_clicked()
{
    on_pushButtonStop_clicked();
    on_stopButton_clicked();
    imageLoaded = true;
    videoLoaded = false;
    ui->playButton->setVisible(false);
    ui->stopButton->setVisible(false);
    ui->videoSlider->setVisible(false);

    if (ui->pushButtonWebcam->isChecked()) {
        ui->pushButtonWebcam->setChecked(false);
    } else {
        ui->checkBoxContinuous->setCheckState(Qt::Unchecked);
        ui->checkBoxContinuous->setEnabled(false);
        continuousMode = false;
    }

    outputTensor.clear();
    ui->labelInference->setText(inferenceTimeLabel);

    imageToSend = imageNew;
    matToSend = matNew;
    drawMatToView(matToSend);
}

void MainWindow::on_pushButtonWebcam_clicked()
{
    on_stopButton_clicked();
    imageLoaded = true;
    videoLoaded = false;
    ui->playButton->setVisible(false);
    ui->stopButton->setVisible(false);
    ui->videoSlider->setVisible(false);
    outputTensor.clear();
    ui->labelInference->setText(inferenceTimeLabel);
    fpsTimer->start();
    if (ui->pushButtonWebcam->isChecked())
        QMetaObject::invokeMethod(cvWorker, "readFrame");
    else
        webcamTimer->stop();
}

void MainWindow::showImage(const cv::Mat& matToShow)
{
    if (ui->pushButtonWebcam->isChecked()) {
        matNew = matToShow;
        QMetaObject::invokeMethod(cvWorker, "readFrame");
        webcamTimer->start();
        matToSend = matNew;
    }

    if (ui->pushButtonWebcam->isChecked() && !continuousMode) {
        drawMatToView(matNew);
        drawFPS(fpsTimer->restart());
        drawBoxes();
    }
}

void MainWindow::drawBoxes()
{
    for (int i = 0; (i + 5) < outputTensor.size(); i += 6) {
        QPen pen;
        QBrush brush;
        QGraphicsTextItem* itemName = scene->addText(nullptr);
        float ymin = outputTensor[i + 2] * float(scene->height());
        float xmin = outputTensor[i + 3] * float(scene->width());
        float ymax = outputTensor[i + 4] * float(scene->height());
        float xmax = outputTensor[i + 5] * float(scene->width());
        float scorePercentage = outputTensor[i + 1] * 100;

        pen.setColor(BOX_COLOUR);
        pen.setWidth(BOX_WIDTH);

        scene->addRect(double(xmin), double(ymin), double(xmax - xmin), double(ymax - ymin), pen, brush);

        itemName->setHtml(QString("<div style='background:rgba(0, 0, 0, 100%);font-size:x-large;'>" + \
                                  QString(labelList[int(outputTensor[i])] + " " + \
                                  QString::number(double(scorePercentage), 'f', 1) + "%") + \
                          QString("</div>")));
        itemName->setPos(double(xmin - X_TEXT_OFFSET), double(ymin - Y_TEXT_OFFSET));
        itemName->setDefaultTextColor(TEXT_COLOUR);
        itemName->setZValue(1);
    }

}

void MainWindow::drawFPS(qint64 timeElapsed)
{
    float fpsValue = 1000.0/timeElapsed;
    QGraphicsTextItem* itemFPS = scene->addText(nullptr);
    itemFPS->setHtml(QString("<div style='background:rgba(0, 0, 0, 100%);font-size:x-large;'>" + \
                      QString( QString::number(double(fpsValue), 'f', 1) + " FPS") + \
                      QString("</div>")));
    itemFPS->setPos(scene->width() - X_FPS , Y_FPS);
    itemFPS->setDefaultTextColor(TEXT_COLOUR);
    itemFPS->setZValue(1);
}

/*
 * If the webcam button is clicked
 * and the webcam button is checked (pressed down),
 * then enable the webcam continuous checkbox.
 */
void MainWindow::pushButtonWebcamCheck(bool webcamButtonChecked)
{
    if (webcamButtonChecked) {
        ui->checkBoxContinuous->setEnabled(true);
        videoLoaded = false;
        imageLoaded = true;
    } else {
        webcamTimer->stop();
        ui->checkBoxContinuous->setCheckState(Qt::Unchecked);
        ui->checkBoxContinuous->setEnabled(false);
        continuousMode = false;
    }
}

void MainWindow::webcamInitStatus(bool webcamStatus)
{
    if (!webcamStatus) {
        webcamTimer->stop();
        ui->pushButtonWebcam->setEnabled(false);
        ui->pushButtonCapture->setEnabled(false);
        ui->pushButtonWebcam->setChecked(false);
	ui->labelCamera->setText(cameraStatusLabel + QString("Disconnected"));
    } else {
        ui->pushButtonWebcam->setEnabled(true);
        ui->pushButtonCapture->setEnabled(true);
        QMetaObject::invokeMethod(cvWorker, "readFrame");
        webcamTimer->setInterval(1000);
        webcamTimer->setSingleShot(true);
        connect(webcamTimer, SIGNAL(timeout()), this, SLOT(webcamTimeout()));
	ui->labelCamera->setText(cameraStatusLabel + QString("Connected"));
    }
}

void MainWindow::on_actionLicense_triggered()
{
    QMessageBox::information(this, "License",
                             "Copyright (C) 2020 Renesas Electronics Corp.\n\n" \
                             "The RZG Object Detection Demo is free software using the Qt Open Source Model: "\
                             "you can redistribute it and/or modify "\
                             "it under the terms of the GNU General Public License as published by "\
                             "the Free Software Foundation, either version 2 of the License, or "\
                             "(at your option) any later version.\n\n" \
                             "The RZG Object Detection Demo is distributed in the hope that it will be useful, "\
                             "but WITHOUT ANY WARRANTY; without even the implied warranty of "\
                             "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "\
                             "GNU General Public License for more details.\n\n" \
                             "You should have received a copy of the GNU General Public License "\
                             "along with the RZG Object Detection Demo. If not, see https://www.gnu.org/licenses.");
}

void MainWindow::on_actionReset_triggered()
{
    QMetaObject::invokeMethod(cvWorker, "initialiseWebcam", Qt::AutoConnection, Q_ARG(QString,webcamName));
}

void MainWindow::webcamTimeout()
{
    opencvThread->deleteLater();
    ui->pushButtonWebcam->setEnabled(false);
    ui->pushButtonCapture->setEnabled(false);
    ui->pushButtonWebcam->setChecked(false);
    ui->labelCamera->setText(cameraStatusLabel + QString("Disconnected"));
}

void MainWindow::on_actionDisconnect_triggered()
{
    webcamTimer->stop();
    QMetaObject::invokeMethod(cvWorker, "disconnectWebcam");
    ui->pushButtonWebcam->setEnabled(false);
    ui->pushButtonCapture->setEnabled(false);
    ui->labelCamera->setText(cameraStatusLabel + QString("Disconnected"));
}

void MainWindow::on_playButton_clicked()
{
    if (ui->playButton->isChecked()) {
        ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
        ui->pushButtonWebcam->setChecked(false);
        outputTensor.clear();
        ui->labelInference->setText(inferenceTimeLabel);
        videoTimer->start(fpsToDelay(cap.get(cv::CAP_PROP_FPS)));
        getVideoFileFrame();
    } else {
        on_pushButtonStop_clicked();
    }

}

void MainWindow::on_stopButton_clicked()
{
    on_pushButtonStop_clicked();
    videoTimer->stop();
    videoLoaded = false;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    getVideoFileFrame();
    ui->playButton->setChecked(false);
    ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    scene->clear();
}

void MainWindow::on_checkBoxContinuous_clicked()
{
    if (!ui->checkBoxContinuous->isChecked()) {
        ui->pushButtonRun->setEnabled(true);
        if(videoLoaded && continuousMode) {
            ui->playButton->setChecked(false);
            on_playButton_clicked();
        }
    }
}

void MainWindow::sliderValueChanged(int value)
{
    if (ui->videoSlider->isSliderDown()) {
        on_pushButtonStop_clicked();
        cap.set(cv::CAP_PROP_POS_FRAMES, value);
    }
}

void MainWindow::on_videoSlider_sliderReleased()
{
    matToSend = captureVideoFrame();
    drawMatToView(matToSend);
    drawBoxes();
}

cv::Mat MainWindow::resizeKeepAspectRatio(const cv::Mat& matInput)
{
    cv::Mat matOutput;

    double height = ui->graphicsView->width() * (matInput.rows/(double)matInput.cols);
    double width = (ui->graphicsView->height() - HEIGHT_OFFSET) * (matInput.cols/(double)matInput.rows);

    if( height <= (ui->graphicsView->height() - HEIGHT_OFFSET))
        cv::resize(matInput, matOutput, cv::Size(ui->graphicsView->width(), height));
    else
        cv::resize(matInput, matOutput, cv::Size(width, (ui->graphicsView->height() - HEIGHT_OFFSET)));

    return matOutput;
}

cv::Mat MainWindow::captureVideoFrame()
{
    if (cap.get(cv::CAP_PROP_POS_FRAMES) == cap.get(cv::CAP_PROP_FRAME_COUNT))
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    cv::Mat videoFrame;
    cap >> videoFrame;
    if (videoFrame.empty())
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    return videoFrame;
}

QImage MainWindow::matToQImage(const cv::Mat& matToConvert)
{
    cv::Mat matToConvertRGB;
    QImage convertedImage;

    if (matToConvert.empty())
        return QImage(nullptr);

    cv::cvtColor(matToConvert, matToConvertRGB, cv::COLOR_BGR2RGB);

    convertedImage = QImage(matToConvertRGB.data, matToConvertRGB.cols, \
                     matToConvertRGB.rows, int(matToConvertRGB.step), \
                        QImage::Format_RGB888).copy();
    return convertedImage;
}

void MainWindow::getVideoFileFrame()
{
        cv::Mat videoMat;
        ui->videoSlider->setValue(cap.get(cv::CAP_PROP_POS_FRAMES));
        videoMat = captureVideoFrame();

        if (!videoLoaded) {
            matToSend = videoMat;
            videoLoaded = true;
        }

        if (ui->playButton->isChecked()) {
            ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
            matToSend = videoMat;
        }

        if ((ui->playButton->isChecked() && !continuousMode) || !videoLoaded) {
            drawMatToView(videoMat);
            drawFPS(fpsTimer->restart());
            drawBoxes();
        }
}

int MainWindow::fpsToDelay(float fps)
{
    return 1000/fps;
}

void MainWindow::drawMatToView(const cv::Mat& matInput)
{
    cv::Mat matToDraw;
    QImage imageToDraw;

    matToDraw = resizeKeepAspectRatio(matInput);
    imageToDraw = matToQImage(matToDraw);

    image = QPixmap::fromImage(imageToDraw);
    scene->clear();
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
}

void MainWindow::on_actionExit_triggered()
{
    QApplication::quit();
}
