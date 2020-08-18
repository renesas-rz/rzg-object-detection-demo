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
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/videoio.hpp>

#define TPU_MODEL_NAME "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
#define CPU_MODEL_NAME "mobilenet_ssd_v2_coco_quant_postprocess.tflite"

#define BOX_WIDTH 4
#define BOX_COLOUR Qt::green
#define TEXT_COLOUR Qt::white
#define X_TEXT_OFFSET 6
#define Y_TEXT_OFFSET 18
#define X_FPS 95              // Place FPS text in top right hand corner
#define Y_FPS 0

Q_DECLARE_METATYPE(cv::Mat)

class QGraphicsScene;
class QGraphicsView;
class QEventLoop;
class opencvWorker;
class tfliteWorker;
class QElapsedTimer;

namespace Ui { class MainWindow; } //Needed for mainwindow.ui

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent, QString cameraLocation, QString labelLocation, \
               QString modelLocation, bool tpuEnable);

signals:
    void sendImage(const cv::Mat&);
    void sendFrameImage(const cv::Mat&);
    void sendNumOfInferenceThreads(int threads);
    void fileLoaded();
    void sendInitWebcam();
    void sendReadFrame();
    void processImage();


public slots:
    void receiveRequest();
    void showImage(const cv::Mat& matToShow);
    void pushButtonWebcamCheck(bool webcamButtonChecked);
    void sliderValueChanged(int value);


private slots:
    void receiveOutputTensor (const QVector<float>& receivedTensor, int recievedTimeElapsed, const cv::Mat &receivedMat);
    void webcamInitStatus (bool webcamStatus);
    void on_pushButtonFile_clicked();
    void on_pushButtonRun_clicked();
    void on_inferenceThreadCount_valueChanged(int value);
    void on_pushButtonStop_clicked();
    void on_pushButtonCapture_clicked();
    void on_pushButtonWebcam_clicked();
    void on_actionLicense_triggered();
    void on_actionReset_triggered();
    void webcamTimeout();
    void on_actionDisconnect_triggered();
    void on_playButton_clicked();
    void on_stopButton_clicked();
    void on_checkBoxContinuous_clicked();
    void on_videoSlider_sliderReleased();
    void getVideoFileFrame();
    void on_actionExit_triggered();

private:
    void drawBoxes();
    void drawFPS(qint64 timeElapsed);
    void drawMatToView(const cv::Mat& matInput);
    int fpsToDelay (float fps);
    cv::Mat captureVideoFrame();
    QImage matToQImage(const cv::Mat& matToConvert);

    Ui::MainWindow *ui;
    QPixmap image;
    QGraphicsScene *scene;
    QImage imageNew;
    QImage imageToSend;
    QVector<float> outputTensor;
    QGraphicsView *graphicsView;
    opencvWorker *cvWorker;
    tfliteWorker *tfWorker;
    QThread *opencvThread, *tfliteThread;
    QEventLoop *qeventLoop;
    QTimer *webcamTimer;
    QString webcamName;
    QStringList labelList;
    QString inferenceTimeLabel;
    bool imageLoaded;
    bool videoLoaded;
    bool continuousMode;
    QElapsedTimer *fpsTimer;
    QTimer *videoTimer;
    QElapsedTimer *videoContinuousTimer;
    cv::VideoCapture cap;
    cv::Mat matToSend;
    cv::Mat matNew;
};

#endif // MAINWINDOW_H
