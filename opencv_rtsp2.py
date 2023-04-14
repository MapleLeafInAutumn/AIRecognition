import cv2

from datetime import datetime
import ffmpeg
import numpy as np
import cv2


def main(source):
    args = {"rtsp_transport": "tcp"}    # 添加参数
    probe = ffmpeg.probe(source)
    cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    print("fps: {}".format(cap_info['r_frame_rate']))
    width = cap_info['width']           # 获取视频流的宽度
    height = cap_info['height']         # 获取视频流的高度
    up, down = str(cap_info['r_frame_rate']).split('/')
    fps = eval(up) / eval(down)
    print("fps: {}".format(fps))    # 读取可能会出错错误
    process1 = (
        ffmpeg
        .input(source, **args)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )
    while True:
        in_bytes = process1.stdout.read(width * height * 3)     # 读取图片
        if not in_bytes:
            break
        # 转成ndarray
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
        cv2.putText(in_frame, dt_string, (210, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.imshow("ffmpeg", in_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    process1.kill()

if __name__ == "__main__":
    # rtsp流需要换成自己的
    source = "rtsp://47.243.7.221/car004"
    main(source)


