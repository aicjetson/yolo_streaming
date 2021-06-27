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

        # ラベルのデータダウンロード
        self.categories = download_label()

        # YOLO物体検出結果の後処理のパラメータを指定 
        self.postprocessor_args = {
            # YOLO v2 マスク
            "yolo_masks": [(0, 1, 2, 3, 4)],
            # YOLOv2 アンカーボックスの情報
            "yolo_anchors": [
                (1.08, 1.19),
                (3.42, 4.41),
                (6.63, 11.38),
                (9.42, 5.11),
                (16.62, 10.52),
            ],
            # 物体だと判断するスコア基準
            "obj_threshold": 0.6,
            # 非最大値抑制に用いる閾値
            "nms_threshold": 0.3,
            # 入寮画像解像度
            "yolo_input_resolution": INPUT_RES,
            # 検出対象になる物体カテゴリの数
            "num_categories": len(self.categories),
        }
        self.postprocessor = PostprocessYOLO(**self.postprocessor_args)

        # 後処理に入力するデータの形
        self.output_shapes = [(1, 125, 13, 13)]

        # ONNXモデルをダウンロード
        self.onnx_file_path = download_model()

        # ローカルに保存するTensor RT Planファイル名前を指定
        self.engine_file_path = "model.trt"

        self.time_list = np.zeros(10)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # read()は、二つの値を返すので、success, imageの2つ変数
        # OpencVはデフォルトでは raw imagesなので JPEGに変換
        # メモリ上に格納したい時はimencodeを使用
        success, image = self.video.read()
        ret, frame = cv2.imencode(".jpg", image)
        return frame

    def get_detection(
        self, inputs, outputs, bindings, stream, fps, frame_count, context
    ):

        # カメラからフレームキャプチャ
        ret, img = self.video.read()

        # Tiny YOLO v2入力にあわせてフレームを変換
        rs_img = cv2.resize(img, INPUT_RES)
        rs_img = cv2.cvtColor(rs_img, cv2.COLOR_BGRA2RGB)
        src_img = reshape_image(rs_img)

        # TensorRTを使って推論実行する
        inputs[0].host = src_img
        trt_outputs = common.do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        # 後処理用に出力データをネットワークの出力データ形状を変換する．
        trt_outputs = [
            output.reshape(shape)
            for output, shape in zip(trt_outputs, self.output_shapes)
        ]

        # 後処理を実行してBBOXを計算する．
        boxes, classes, scores = self.postprocessor.process(
            trt_outputs, (self.act_width, self.act_height)
        )

        # BBOXを画像に描画する．
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
