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

#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>

#include "tfliteworker.h"

tfliteWorker::tfliteWorker(bool tpuEnable, QString modelLocation)
{
    tflite::ops::builtin::BuiltinOpResolver tfliteResolver;
    TfLiteIntArray *wantedDimensions;

    if (tpuEnable) {
        numberOfInferenceThreads = 1;
        edgetpu::EdgeTpuContext* edgetpu_context =
              edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext().release();
        tfliteResolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
        tfliteModel = tflite::FlatBufferModel::BuildFromFile(modelLocation.toStdString().c_str());
        tflite::InterpreterBuilder(*tfliteModel, tfliteResolver) (&tfliteInterpreter);
        tfliteInterpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
    } else {
        numberOfInferenceThreads = 2;
        tfliteModel = tflite::FlatBufferModel::BuildFromFile(modelLocation.toStdString().c_str());
        tflite::InterpreterBuilder(*tfliteModel, tfliteResolver) (&tfliteInterpreter);
    }

    if (tfliteInterpreter->AllocateTensors() != kTfLiteOk && tpuEnable)
        qFatal("Failed to allocate tensors, is the TPU device connected?");

    tfliteInterpreter->SetProfiler(nullptr);
    tfliteInterpreter->SetNumThreads(numberOfInferenceThreads);

    wantedDimensions = tfliteInterpreter->tensor(tfliteInterpreter->inputs()[0])->dims;
    wantedHeight = wantedDimensions->data[1];
    wantedWidth = wantedDimensions->data[2];
    wantedChannels = wantedDimensions->data[3];
}

void tfliteWorker::process()
{
    tfliteInterpreter->SetNumThreads(numberOfInferenceThreads);
    emit requestImage();
}

void tfliteWorker::receiveImage(const cv::Mat& sentMat)
{
    cv::Mat sentImageMat;
    std::chrono::high_resolution_clock::time_point startTime, stopTime;
    int timeElapsed;
    int input = tfliteInterpreter->inputs()[0];

    cv::resize(sentMat, sentImageMat, cv::Size(wantedHeight,wantedWidth));
    cv::cvtColor(sentImageMat, sentImageMat, cv::COLOR_BGR2RGB);

    memcpy(tfliteInterpreter->typed_tensor<uint8_t>(input), sentImageMat.data, sentImageMat.total() * sentImageMat.elemSize());

    startTime = std::chrono::high_resolution_clock::now();

    tfliteInterpreter->Invoke();

    stopTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; tfliteInterpreter->typed_output_tensor<float>(2)[i] > float(DETECT_THRESHOLD)\
         && tfliteInterpreter->typed_output_tensor<float>(2)[i] <= float(1.0); i++){
        outputTensor.push_back(tfliteInterpreter->typed_output_tensor<float>(1)[i]);          //item
        outputTensor.push_back(tfliteInterpreter->typed_output_tensor<float>(2)[i]);          //confidence
        outputTensor.push_back(tfliteInterpreter->typed_output_tensor<float>(0)[i * 4]);      //box ymin
        outputTensor.push_back(tfliteInterpreter->typed_output_tensor<float>(0)[i * 4 + 1]);  //box xmin
        outputTensor.push_back(tfliteInterpreter->typed_output_tensor<float>(0)[i * 4 + 2]);  //box ymax
        outputTensor.push_back(tfliteInterpreter->typed_output_tensor<float>(0)[i * 4 + 3]);  //box xmax
    }

    timeElapsed = int(std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count());
    emit sendOutputTensor(outputTensor, timeElapsed, sentMat);
    outputTensor.clear();
}

void tfliteWorker::receiveNumOfInferenceThreads(int threads)
{
    numberOfInferenceThreads = threads;
}
