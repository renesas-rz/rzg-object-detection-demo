#*****************************************************************************************
# Copyright (C) 2020 Renesas Electronics Corp.
# This file is part of the RZG Object Detection Demo.
#
# The RZG Object Detection Demo is free software using the Qt Open Source Model: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# The RZG Object Detection Demo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the RZG Object Detection Demo.  If not, see <https://www.gnu.org/licenses/>.
#*****************************************************************************************

QT += core gui multimedia widgets

CONFIG += c++11

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    opencvworker.cpp \
    tfliteworker.cpp

HEADERS += \
    mainwindow.h \
    opencvworker.h \
    tfliteworker.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += \
    $$(SDKTARGETSYSROOT)/usr/include/opencv4 \
    $$(SDKTARGETSYSROOT)/usr/include/tensorflow/lite/tools/make/downloads/flatbuffers/include

LIBS += \
    -L $$(SDKTARGETSYSROOT)/usr/lib64 \
    -lopencv_core \
    -lopencv_videoio \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -ltensorflow-lite \
    -ldl \
    -lutil \
    -l:libedgetpu.so.1
