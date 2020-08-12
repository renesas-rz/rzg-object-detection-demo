# RZ/G Object Detection Demo Application Source Code

This repository contains the code required to build the demo application.  
This demo is based upon the [Renesas RZ/G AI BSP](https://github.com/renesas-rz/meta-renesas-ai)
and requires [TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/v2.0.0) and [OpenCV](https://opencv.org/).

## Manual Build Instructions
### Target: RZ/G2M (hihope-rzg2m)
1. Have a suitable cross toolchain by building `bitbake core-image-qt-sdk -c populate_sdk`
with the Yocto meta-layers described in [meta-rzg2](https://github.com/renesas-rz/meta-rzg2)
and [meta-renesas-ai](https://github.com/renesas-rz/meta-renesas-ai) (copy `.conf` files from meta-tensorfow-lite).
2. Install cross toolchain with `sudo sh ./poky-glibc-x86_64-core-image-qt-sdk-aarch64-toolchain-2.4.3.sh`.
3. Set up environment variables with `source /<SDK location>/environment-setup-aarch64-poky-linux`.
4. Run `qmake`.
5. Run `make`.
6. Copy `object_detection_demo` to the root filesystem.
7. Copy [`coco_labels.txt`](https://github.com/google-coral/edgetpu/blob/master/test_data/coco_labels.txt) to the directory where `object_detection_demo` is located.
8. Copy [`mobilenet_ssd_v2_coco_quant_postprocess.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite) to the directory where `object_detection_demo` is located.
9. Copy [`mobilenet_ssd_v2_coco_quant_postprocess_tpu.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite) to the directory where `object_detection_demo` is located.
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
    ```

2. Build and install [TensorFlow Lite v2.0.0](https://github.com/tensorflow/tensorflow/tree/v2.0.0).
    ```
    git clone  https://github.com/tensorflow/tensorflow.git
    cd tensorflow/
    git checkout v2.0.0
    ./tensorflow/lite/tools/make/download_dependencies.sh
    ./tensorflow/lite/tools/make/build_lib.sh
    sudo cp ./tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a /usr/local/lib/
    sudo cp -r tensorflow/ /usr/local/include
    sudo cp -r tensorflow/lite/tools/make/downloads/flatbuffers/include/flatbuffers/ /usr/local/include
    ```
3. Run `sudo apt install qt5-default`
4. Run `sudo apt install qtmultimedia5-dev`
5. Run `qmake`
6. Run `make` 
7. Copy [`coco_labels.txt`](https://github.com/google-coral/edgetpu/blob/master/test_data/coco_labels.txt) to the directory where `object_detection_demo` is located.
8. Copy [`mobilenet_ssd_v2_coco_quant_postprocess.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess.tflite)` to the directory where `object_detection_demo` is located.
9. Copy [`mobilenet_ssd_v2_coco_quant_postprocess_tpu.tflite`](https://github.com/google-coral/edgetpu/blob/diploria2/test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite) to the directory where `object_detection_demo` is located.
10. Run the app with `./object_detection_demo`.

## Help
```
Usage: ./object_detection_demo [options]
Object Detection Demo
  Draws boxes around detected objects and displays the name and
  confidence of the object. Also displays inference time.

Required Hardware:
  Camera: Currently supports Logitech C922 Pro Stream, should
          work with any UVC compatible USB camera that has a
          supported resolution of 800x600.
  TPU Mode: Requires Coral USB Accelerator.

Supported Models:
  MobileNet v2 SSD Quantised TensorFlow Lite

Buttons:
  Run: Run inference on the selected image once.
  Load Image: Load an image from the filesystem. Supported formats are
              bmp, jpg, and png.
  Load Webcam: Load a webcam stream.
  Capture Image: Capture an image from the webcam.
  Continuous Checkbox: Only available when a webcam stream is loaded.
                       Enable to continuously run inference.
  Stop: Stop continuous inference.
  Threads: Only available in CPU mode. Change the number of inference
           threads.
  About->License: Read the license that this app is licensed under.
  Camera->Reset: Reset the connection to the webcam.
  Camera->Disconnect: Disconnect the currently connected webcam.

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
