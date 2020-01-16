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

#define TPU_MODEL_NAME "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
#define CPU_MODEL_NAME "mobilenet_ssd_v2_coco_quant_postprocess.tflite"

#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 600

#define MAINWINDOW_WIDTH 1280
#define MAINWINDOW_HEIGHT 720
#define TABLE_COLUMN_WIDTH 180
#define BOX_WIDTH 4
#define BOX_COLOUR Qt::green
#define TEXT_COLOUR Qt::white
#define X_TEXT_OFFSET 6
#define Y_TEXT_OFFSET 18

class QGraphicsScene;
class QGraphicsView;
class QEventLoop;
class opencvWorker;
class tfliteWorker;

namespace Ui { class MainWindow; } //Needed for mainwindow.ui

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent, QString cameraLocation, QString labelLocation, \
               QString modelLocation, bool tpuEnable);

signals:
    void sendImage(const QImage&);
    void sendNumOfInferenceThreads(int threads);
    void imageLoaded();
    void sendInitWebcam();
    void sendReadFrame();
    void processImage();

public slots:
    void receiveRequest();
    void showImage(const QImage& imageToShow);
    void pushButtonWebcamCheck(bool webcamButtonChecked);


private slots:
    void receiveOutputTensor (const QVector<float>& receivedTensor, int recievedTimeElapsed);
    void webcamInitStatus (bool webcamStatus);
    void on_pushButtonImage_clicked();
    void on_pushButtonRun_clicked();
    void on_inferenceThreadCount_valueChanged(int value);
    void on_pushButtonStop_clicked();
    void on_pushButtonCapture_clicked();
    void on_pushButtonWebcam_clicked();
    void on_checkBoxContinuous_stateChanged(int checkBoxState);
    void on_actionLicense_triggered();
    void on_actionReset_triggered();
    void webcamTimeout();
    void on_actionDisconnect_triggered();

private:
    void drawBoxes();

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
    bool continuousRunning;
    bool stopClicked;

};

#endif // MAINWINDOW_H
