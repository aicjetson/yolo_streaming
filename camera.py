import cv2
import numpy as np

import common
from data_processing import PostprocessYOLO
from yolo_tool import (
    draw_bboxes,
    draw_message,
    reshape_image,
    download_label,
    download_model,
)
from utils import Singleton

INPUT_RES = (416, 416)


class Camera(Singleton):
    def __init__(self):
        self.width = 1920
        self.height = 1080
        self.video = cv2.VideoCapture(0, cv2.CAP_V4L)

        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.act_width = self.width
        self.act_height = self.height
        self.frame_info = "Frame:%dx%d" % (self.act_width, self.act_height)

        # Download the label data
        self.categories = download_label()

        # Configure the post-processing
        self.postprocessor_args = {
            # YOLO masks (Tiny YOLO v2 has only single scale.)
            "yolo_masks": [(0, 1, 2, 3, 4)],
            # YOLO anchors
            "yolo_anchors": [
                (1.08, 1.19),
                (3.42, 4.41),
                (6.63, 11.38),
                (9.42, 5.11),
                (16.62, 10.52),
            ],
            # Threshold of object confidence score (between 0 and 1)
            "obj_threshold": 0.6,
            # Threshold of NMS algorithm (between 0 and 1)
            "nms_threshold": 0.3,
            # Input image resolution
            "yolo_input_resolution": INPUT_RES,
            # Number of object classes
            "num_categories": len(self.categories),
        }
        self.postprocessor = PostprocessYOLO(**self.postprocessor_args)

        # Image shape expected by the post-processing
        self.output_shapes = [(1, 125, 13, 13)]
        self.onnx_file_path = download_model()

        self.engine_file_path = "model.trt"

        self.time_list = np.zeros(10)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, frame = cv2.imencode(".jpg", image)
        return frame

        # read()は、二つの値を返すので、success, imageの2つ変数で受けています。
        # OpencVはデフォルトでは raw imagesなので JPEGに変換
        # ファイルに保存する場合はimwriteを使用、メモリ上に格納したい時はimencodeを使用
        # cv2.imencode() は numpy.ndarray() を返すので .tobytes() で bytes 型に変換

    def get_detection(
        self, inputs, outputs, bindings, stream, fps, frame_count, context
    ):
        # Get the frame start time for FPS calculation

        # Capture a frame
        ret, img = self.video.read()

        # Reshape the capture image for Tiny YOLO v2
        rs_img = cv2.resize(img, INPUT_RES)
        rs_img = cv2.cvtColor(rs_img, cv2.COLOR_BGRA2RGB)
        src_img = reshape_image(rs_img)

        # Execute an inference in TensorRT
        inputs[0].host = src_img
        trt_outputs = common.do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        # Reshape the network output for the post-processing
        trt_outputs = [
            output.reshape(shape)
            for output, shape in zip(trt_outputs, self.output_shapes)
        ]

        # Calculates the bounding boxes
        boxes, classes, scores = self.postprocessor.process(
            trt_outputs, (self.act_width, self.act_height)
        )

        # Draw the bounding boxes
        if boxes is not None:
            draw_bboxes(img, boxes, scores, classes, self.categories)
        if frame_count > 10:
            fps_info = "{0}{1:.2f}".format("FPS:", fps)
            msg = "%s %s" % (self.frame_info, fps_info)
            draw_message(img, msg)
        if frame_count == 10:
            cv2.imwrite("./img.jpg", img)

        ret, frame = cv2.imencode(".jpg", img)

        return frame
