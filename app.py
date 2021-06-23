import numpy as np
import time
from flask import Flask, render_template, Response

import pycuda.driver as cuda

import common
from camera import Camera
from get_engine import get_engine

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return render_template("stream.html")


def gen_video(camera):
    # TensorRTにモデルを読み込む
    with get_engine(
        camera.onnx_file_path, camera.engine_file_path
    ) as engine, engine.create_execution_context() as context:

        # TensorRT用にバッファメモリを割り当て
        ctx = cuda.Context.attach()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        ctx.detach()

        fps = 0.0
        frame_count = 0

        while True:
            # フレームの開始時刻を記録
            start_time = time.time()

            # 物体検出が完了してBBOXや各種情報が書き込まれた画像を取得する．
            frame = camera.get_detection(
                inputs, outputs, bindings, stream, fps, frame_count, context
            )

            # フレームのイテレータの作成
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
                )
            else:
                print("frame is none")

            # 1フレームの処理時間を計測してFPS値を算出する
            elapsed_time = time.time() - start_time
            time_list = np.append(camera.time_list, elapsed_time)
            time_list = np.delete(time_list, 0)
            avg_time = np.average(time_list)
            fps = 1.0 / avg_time

            frame_count += 1


@app.route("/video_feed")
def video_feed():
    cam = Camera()
    return Response(
        gen_video(cam), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/kill_camera")
def kill_camera():
    cam = Camera()
    del cam
    return render_template("camera_killed.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
