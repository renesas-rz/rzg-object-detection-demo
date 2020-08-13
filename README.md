# RZ/G Object Detection Demo Application Source Code

This repository contains the code required to build the demo application.
This demo is based upon the [Renesas RZ/G AI BSP](https://github.com/renesas-rz/meta-renesas-ai)
and requires [TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/v2.0.2) and [OpenCV](https://opencv.org/).

## Manual Build Instructions
### Target: RZ/G2M (hihope-rzg2m)
1. Setup the the Yocto environment as described in [meta-renesas-ai](https://github.com/renesas-rz/meta-renesas-ai) with the meta-layers described in [meta-rzg2](https://github.com/renesas-rz/meta-rzg2).
   ```
   Copy the local.conf file from meta-renesas-ai/meta-benchmark/templates/<Board>/local.conf into the build/conf directory.
   Build a suitable cross toolchain with `bitbake core-image-qt-sdk -c populate_sdk`.
   ```
2. Install cross toolchain with `sudo sh ./poky-glibc-x86_64-core-image-qt-sdk-aarch64-toolchain-2.4.3.sh`.
3. Set up environment variables with `source /<SDK location>/environment-setup-aarch64-poky-linux`.
4. Run `qmake`.
5. Run `make`.
6. Copy `object_detection_demo` to the root filesystem.
7. Checkout and copy [`coco_labels.txt`](https://github.com/google-coral/edgetpu/blob/master/test_data/coco_labels.txt) to the directory where `object_detection_demo` is located.
8. Checkout and copy [`mobilenet_ssd_v2_coco_quant_postprocess.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite) to the directory where `object_detection_demo` is located.
9. Checkout and copy [`mobilenet_ssd_v2_coco_quant_postprocess_tpu.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite) to the directory where `object_detection_demo` is located.
10. Run the app with `./object_detection_demo`.

### Target: Ubuntu
1. Install opencv core and opencv videoio, make sure your version has Gstreamer enabled. Otherwise build and install [OpenCV](https://github.com/opencv/opencv.git).
    ```
    git clone  https://github.com/opencv/opencv.git
    cd opencv/
    mkdir build/
    cd build/
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_V4L=ON \
    -D WITH_QT=ON \
    -D WITH_GSTREAMER=ON ..
    make
    make install
    cd /usr/local/include/opencv4/
    sudo cp -R opencv2/ ../
    ```

2. Build and install [TensorFlow Lite v2.0.2](https://github.com/tensorflow/tensorflow/tree/v2.0.2).
    ```
    git clone  https://github.com/tensorflow/tensorflow.git
    cd tensorflow/
    git checkout v2.0.0
    ./tensorflow/lite/tools/make/download_dependencies.sh
    ./tensorflow/lite/tools/make/build_lib.sh
    sudo cp ./tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a /usr/local/lib/
    sudo cp -r tensorflow/ /usr/local/include
    sudo cp -r tensorflow/lite/tools/make/downloads/flatbuffers/include/flatbuffers/ /usr/local/include
    ```
3. Install Google Coral
    ```
    git clone https://github.com/google-coral/edgetpu.git
    cd edgetpu
    git checkout diploria2
    sudo ./scripts/runtime/install.sh
    sudo mkdir /usr/include/google-coral-diploria2
    sudo cp libedgetpu/edgetpu.h /usr/include/google-coral-diploria2
    ```
4. Run `sudo apt install qt5-default`
5. Run `sudo apt install qtmultimedia5-dev`
6. Compilation is then possible with QtCreator.
8. Checkout and copy [`coco_labels.txt`](https://github.com/google-coral/edgetpu/blob/master/test_data/coco_labels.txt) to the directory where `object_detection_demo` is located.
9. Checkout and copy [`mobilenet_ssd_v2_coco_quant_postprocess.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite)` to the directory where `object_detection_demo` is located.
10. Checkout and copy [`mobilenet_ssd_v2_coco_quant_postprocess_tpu.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite) to the directory where `object_detection_demo` is located.
11. Run the app with `./object_detection_demo`.

## Help
```
Usage: ./object_detection_demo [options]
Object Detection Demo
  Draws boxes around detected objects and displays the name and
  confidence of the object. Also displays inference time and FPS
  if applicable.

Required Hardware:
  Camera: Currently supports Logitech C922 Pro Stream, should
          work with any UVC compatible USB camera that has a
          supported resolution of 800x600.
  TPU Mode: Requires Coral USB Accelerator.

Supported Models:
  MobileNet v2 SSD Quantised TensorFlow Lite

Buttons:
  Run: Run inference on the selected image once.
  Load File: Load an image or video from the filesystem. Media formats
             supported by OpenCV 4.1.1 and GStreamer 1.12.2 can be
             opened, including mp4, m4v, mkv, webm, bmp, jpg, png.
  Video Controls: Only available when video is loaded. Play,
                  pause, stop, and seek can be performed.
  Load Camera: Load a camera stream.
  Capture Image: Capture an image from the camera.
  Continuous Checkbox: Only available when a camera stream or video
                       is loaded. Enable to continuously run inference.
  Stop: Stop continuous inference.
  Threads: Only available in CPU mode. Change the number of inference
           threads.
  About->License: Read the license that this app is licensed under.
  Camera->Reset: Reset the connection to the camera.
  Camera->Disconnect: Disconnect the currently connected camera.

Default options:
  Camera: /dev/v4l/by-id/<first file>
  Label: ./*label*.txt
  Model:
    CPU Mode: ./mobilenet_ssd_v2_coco_quant_postprocess.tflite
    TPU Mode: ./mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
  TPU Mode: Disabled

Options:
  -c, --camera <file>  Choose a camera.
  -l, --label <file>   Choose a label file.
  -m, --model <file>   Choose a model.
  -t, --tpu            Enable tpu processing.
  -h, --help           Displays this help.
```
