#! /bin/bash

docker run -it \
--runtime nvidia \
--network host \
-v ~/projects_01/stream_detect:/usr/app \
--device /dev/video0 \
--name flask_yolo_app_con flask_uwsgi_nginx_01:stream
