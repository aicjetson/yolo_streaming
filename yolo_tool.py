from __future__ import print_function

import cv2
import numpy as np
import os
import wget
import tarfile

from data_processing import load_label_categories

# MODEL_URL = 'https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz'
MODEL_URL = "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz"
LABEL_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names"


# 物体検出処理を行った画像にバウンディングボックスを描画する
def draw_bboxes(image, bboxes, confidences, categories, all_categories, message=None):
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        img_height, img_width, _ = image.shape
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(img_width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(img_height, np.floor(y_coord + height + 0.5).astype(int))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)
        info = "{0} {1:.2f}".format(all_categories[category], score)
        cv2.putText(
            image,
            info,
            (right, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        print(info)
    if message is not None:
        cv2.putText(
            image,
            message,
            (32, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


# 文字列を画像に埋め込んで情報を表示する
def draw_message(image, message):
    cv2.putText(
        image,
        message,
        (32, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


# OpenCVで取り込んだ画像を，YOLOのネットワークに入力するために変換を行う
def reshape_image(img):
    # 8ビット整数から32ビット浮動小数点へ
    img = img.astype(np.float32)
    # HWCの順からCHWの順に入れ替え
    img = np.transpose(img, [2, 0, 1])
    # CHWからNHCWに変換
    img = np.expand_dims(img, axis=0)
    # row-major並びに変換
    img = np.array(img, dtype=np.float32, order="C")
    return img


# URLを指定してファイルをダウンロードする．ダウンロードしてある時は行わない．
def download_file_from_url(url):
    file = os.path.basename(url)
    if not os.path.exists(file):
        print("\nDownload from %s" % url)
        wget.download(url)
    return file


# クラスラベルのファイルをダウンロードする
def download_label():
    file = download_file_from_url(LABEL_URL)
    categories = load_label_categories(file)
    num_categories = len(categories)
    assert num_categories == 20
    return categories


# ONNXの学習済みモデルをダウンロードする．
# ダウンロード後はアーカイブファイルを展開する．
def download_model():
    file = download_file_from_url(MODEL_URL)
    tar = tarfile.open(file)
    infs = tar.getmembers()
    onnx_file = None
    for inf in infs:
        f = inf.name
        _, ext = os.path.splitext(f)
        if ext == ".onnx":
            onnx_file = f
            break
    if not os.path.exists(onnx_file):
        tar.extract(onnx_file)
    tar.close()
    return onnx_file
