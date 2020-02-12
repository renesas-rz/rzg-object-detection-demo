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
#include <QImageReader>

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
    this->resize(MAINWINDOW_WIDTH, MAINWINDOW_HEIGHT);
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    labelFile.setFileName(labelLocation);
    if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qFatal("%s could not be opened.", labelLocation.toStdString().c_str());
    }

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
        ui->labelInference->setText("TPU Inference Time: ");
    } else {
        ui->labelInference->setText("CPU Inference Time: ");
    }

    qRegisterMetaType<cv::Mat>();
    opencvThread = new QThread();
    opencvThread->setObjectName("opencvThread");
    opencvThread->start();
    cvWorker = new opencvWorker();
    cvWorker->moveToThread(opencvThread);
    connect(ui->pushButtonWebcam, SIGNAL(toggled(bool)), this, \
            SLOT(pushButtonWebcamCheck(bool)));
    connect(cvWorker, SIGNAL(sendImage(const QImage&)), this, SLOT(showImage(const QImage&)));
    connect(cvWorker, SIGNAL(webcamInit(bool)), this, SLOT(webcamInitStatus(bool)));

    QMetaObject::invokeMethod(cvWorker, "initialiseWebcam", Qt::AutoConnection, Q_ARG(QString,webcamName));

    qRegisterMetaType<QVector<float> >("QVector<float>");
    tfliteThread = new QThread();
    tfliteThread->setObjectName("tfliteThread");
    tfliteThread->start();
    tfWorker = new tfliteWorker(tpuEnable,modelLocation);
    tfWorker->moveToThread(tfliteThread);
    connect(tfWorker, SIGNAL(requestImage()), this, SLOT(receiveRequest()));
    connect(this, SIGNAL(sendImage(const QImage&)), tfWorker, SLOT(receiveImage(const QImage&)));
    connect(tfWorker, SIGNAL(sendOutputTensor(const QVector<float>&, int, const QImage&)), \
            this, SLOT(receiveOutputTensor(const QVector<float>&, int, const QImage&)));
    connect(this, SIGNAL(sendNumOfInferenceThreads(int)), tfWorker, SLOT(receiveNumOfInferenceThreads(int)));

    webcamTimer = new QTimer();
}

void MainWindow::on_pushButtonImage_clicked()
{
    qeventLoop = new QEventLoop;
    QString fileName;
    QStringList fileNames;
    QFileDialog dialog(this);
    QString imageFilter;

    connect(this, SIGNAL(imageLoaded()), qeventLoop, SLOT(quit()));

    on_pushButtonStop_clicked();
    ui->pushButtonWebcam->setChecked(false);
    outputTensor.clear();
    ui->labelInferenceTime->clear();
    dialog.setFileMode(QFileDialog::AnyFile);

    imageFilter = "Images (";
    for (int i = 0; i < QImageReader::supportedImageFormats().count(); i++) {
        imageFilter += "*." + QImageReader::supportedImageFormats().at(i) + " ";
    }
    imageFilter +=")";

    dialog.setNameFilter(imageFilter);
    dialog.setViewMode(QFileDialog::Detail);

    if (dialog.exec())
        fileNames = dialog.selectedFiles();

    if(fileNames.count() > 0)
        fileName = fileNames.at(0);

    if (!fileName.trimmed().isEmpty()) {
        imageToSend.load(fileName);
        if (imageToSend.width() != IMAGE_WIDTH || imageToSend.height() != IMAGE_HEIGHT)
            imageToSend = imageToSend.scaled(IMAGE_WIDTH, IMAGE_HEIGHT, Qt::KeepAspectRatio);
        image = QPixmap::fromImage(imageToSend);
        scene->clear();
        scene->addPixmap(image);
        scene->setSceneRect(image.rect());
    }

    emit imageLoaded();
    qeventLoop->exec();
}

void MainWindow::on_pushButtonRun_clicked()
{
    if (!(image.depth() > 0)) {
        QMessageBox::warning(this, "Warning", "No source selected, please select an image.");
        return;
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
    sendImage(imageToSend);
    tfWorker->moveToThread(tfliteThread);
}

void MainWindow::receiveOutputTensor(const QVector<float>& receivedTensor, int receivedTimeElapsed, const QImage& receivedImage)
{
    if (ui->pushButtonRun->isEnabled())
        return;

    outputTensor = receivedTensor;

    ui->labelInferenceTime->setText(QString("%1 ms").arg(receivedTimeElapsed));

    if (!ui->checkBoxContinuous->isChecked()) {
        ui->pushButtonRun->setEnabled(true);
    } else {
        image = QPixmap::fromImage(receivedImage);
        scene->clear();
        image.scaled(ui->graphicsView->width(), ui->graphicsView->height(), Qt::KeepAspectRatio);
        scene->addPixmap(image);
        scene->setSceneRect(image.rect());
        QMetaObject::invokeMethod(tfWorker, "process");
    }

    drawBoxes();
}

void MainWindow::on_pushButtonStop_clicked()
{
    ui->pushButtonRun->setEnabled(true);
}

void MainWindow::on_pushButtonCapture_clicked()
{
    on_pushButtonStop_clicked();
    ui->pushButtonWebcam->setChecked(false);
    outputTensor.clear();
    ui->labelInferenceTime->clear();

    imageToSend = imageNew;
    image = QPixmap::fromImage(imageToSend);
    scene->clear();
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
}

void MainWindow::on_pushButtonWebcam_clicked()
{
    outputTensor.clear();
    ui->labelInferenceTime->clear();
}

void MainWindow::showImage(const QImage& imageToShow)
{
    if (ui->pushButtonWebcam->isEnabled()) {
        QMetaObject::invokeMethod(cvWorker, "readFrame");
        webcamTimer->start();
    }

    imageNew = imageToShow;

    if ((imageNew.width() != IMAGE_WIDTH || imageNew.height() != IMAGE_HEIGHT) && imageNew.depth() > 0)
        imageNew = imageNew.scaled(IMAGE_WIDTH, IMAGE_HEIGHT);

    if (ui->pushButtonWebcam->isChecked())
        imageToSend = imageNew;

    if (ui->pushButtonWebcam->isChecked() && !ui->checkBoxContinuous->isChecked()) {
        image = QPixmap::fromImage(imageToSend);
        scene->clear();
        scene->addPixmap(image);
        scene->setSceneRect(image.rect());
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

        itemName->setHtml(QString("<div style='background:rgba(0, 0, 0, 100%);'>" + \
                                  QString(labelList[int(outputTensor[i])] + " " + \
                                  QString::number(double(scorePercentage), 'f', 1) + "%") + \
                          QString("</div>")));
        itemName->setPos(double(xmin - X_TEXT_OFFSET), double(ymin - Y_TEXT_OFFSET));
        itemName->setDefaultTextColor(TEXT_COLOUR);
        itemName->setZValue(1);
    }
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
    } else {
        ui->checkBoxContinuous->setCheckState(Qt::Unchecked);
        ui->checkBoxContinuous->setEnabled(false);
    }
}

void MainWindow::webcamInitStatus(bool webcamStatus)
{
    if (!webcamStatus) {
        webcamTimer->stop();
        ui->pushButtonWebcam->setEnabled(false);
        ui->pushButtonCapture->setEnabled(false);
        QMessageBox::warning(this, "Warning", "Webcam not connected");
        ui->pushButtonWebcam->setChecked(false);
    } else {
        ui->pushButtonWebcam->setEnabled(true);
        ui->pushButtonCapture->setEnabled(true);
        QMetaObject::invokeMethod(cvWorker, "readFrame");
        webcamTimer->setInterval(1000);
        webcamTimer->setSingleShot(true);
        connect(webcamTimer, SIGNAL(timeout()), this, SLOT(webcamTimeout()));
        webcamTimer->start();
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
    QMessageBox::warning(this, "Warning", "Webcam not connected");
    ui->pushButtonWebcam->setChecked(false);
}

void MainWindow::on_actionDisconnect_triggered()
{
    webcamTimer->stop();
    QMetaObject::invokeMethod(cvWorker, "disconnectWebcam");
    ui->pushButtonWebcam->setEnabled(false);
    ui->pushButtonCapture->setEnabled(false);
    QMessageBox::warning(this, "Warning", "Webcam not connected");
}
