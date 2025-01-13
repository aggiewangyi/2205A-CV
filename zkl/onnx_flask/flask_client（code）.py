import base64
import json
import requests
import cv2
import numpy as np
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()


#将图片转编码成二进制流
'opencv'
# def cv2_to_base64(image):
#     image = cv2.imencode('.jpg',image)[1]
#     return base64.b64encode(image).decode('utf-8')


'turbojpeg'
def cv2_to_base64(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoded_image_data = jpeg.encode(image_rgb, quality=75)
    return base64.b64encode(encoded_image_data).decode('utf-8')


#将二进制流解码成图片
'opencv'
# def base64_to_cv2(base64_string):
#
#     image_data = base64.b64decode(base64_string)
#     data = np.frombuffer(image_data, np.uint8)
#     image = cv2.imdecode(data, cv2.IMREAD_COLOR)
#
#     return image

'turbojpeg'
def base64_to_cv2(base64_string):

    image_data = base64.b64decode(base64_string)
    image_rgb = jpeg.decode(image_data)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image




if __name__ == '__main__':
    # url = 'http://10.222.25.63:127/post'
    url = 'http://127.0.0.1:2024/post'

    '视频'
    video_path = r"C:\Users\26296\Desktop\v5\yolov5-master\cut.mkv"
    capture = cv2.VideoCapture(video_path)
    while True:
        success,frame = capture.read()
        image = cv2_to_base64(frame)
        data = {"feed": [{"x": image}], "fetch": ["res"]}

    # '图片'
    # image_path = "000000000009.jpg"
    # frame = cv2.imread(image_path)
    # image = cv2_to_base64(frame)
    # data = {"feed": [{"x": image}], "fetch": ["res"]}

        r = requests.post(url=url, data=json.dumps(data))
        json_data = r.json()
        img_bytes = json_data['feed'][0]['img'].encode('utf8')
        label = json_data['feed'][1]['label']
        print("类别:", label)
        frame = base64_to_cv2(img_bytes)
        cv2.imshow('res', frame)
        cv2.waitKey(10)




