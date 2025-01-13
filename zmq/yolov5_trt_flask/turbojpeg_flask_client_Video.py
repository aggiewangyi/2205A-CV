import base64
import json
import requests
import cv2
import numpy as np
import time

#将图片转编码成二进制流
def cv2_to_base64(image):
    # image = cv2.imencode('.jpg',image)[1]
    image = jpeg.encode(image, quality=90)
    return base64.b64encode(image).decode('utf-8')
    # return image


#将二进制流解码成图片
def base64_to_cv2(base64_string):

    image_data = base64.b64decode(base64_string)
    data = np.frombuffer(image_data, np.uint8)
    # image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = jpeg.decode(data)
    return image

from turbojpeg import TurboJPEG
jpeg = TurboJPEG()
if __name__ == '__main__':

    # url = 'http://10.222.25.63:127/post'
    # url = 'http://127.0.0.1:97/post'
    url = 'http://127.0.0.1:6009/post'
    # url = 'http://192.168.33.41:97/post'
    # image_path = "im1.jpg"
    # image_path = "000000000009.jpg"
    # frame = cv2.imread(image_path)

    cap = cv2.VideoCapture("cut.mp4")

    while True:

        start_time = time.time()
        success, frame = cap.read()

        # image = cv2_to_base64(frame)
        image = jpeg.encode(frame, quality=90)

        # data = {"feed":[{"x": image}],"fetch":["res"]}
        data = {"x": image}
        # r = requests.post(url=url,data=json.dumps(data))
        r = requests.post(url=url,files=data)
        json_data = r.json()
        img_bytes = json_data['feed'][0]['img'].encode('utf8')
        label = json_data['feed'][1]['label']
        # print("类别:", label)

        # frame = base64_to_cv2(img_bytes)

        image_data = base64.b64decode(img_bytes)
        data = np.frombuffer(image_data, np.uint8)
        frame = jpeg.decode(data)

        cv2.imshow('res', frame)
        print(f'time:{(time.time() - start_time) * 1000}')
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == 27:
            break






















