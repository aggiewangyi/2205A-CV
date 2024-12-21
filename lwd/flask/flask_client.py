import base64
import json
import requests
import cv2
import numpy as np


#将图片转编码成二进制流
def cv2_to_base64(image):
    image = cv2.imencode('.jpg',image)[1]
    return base64.b64encode(image).decode('utf-8')


#将二进制流解码成图片
def base64_to_cv2(base64_string):

    image_data = base64.b64decode(base64_string)
    data = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return image



if __name__ == '__main__':

    # url = 'http://10.222.25.63:127/post'
    # url = 'http://127.0.0.1:97/post'
    url = 'http://127.0.0.1:6008/post'
    # image_path = "im1.jpg"
    image_path = "./000000000009.jpg"
    frame = cv2.imread(image_path)
    image = cv2_to_base64(frame)
    data = {"feed":[{"x": image}],"fetch":["res"]}
    r = requests.post(url=url,data=json.dumps(data))
    json_data = r.json()
    img_bytes = json_data['feed'][0]['img'].encode('utf8')
    label = json_data['feed'][1]['label']
    print("类别:", label)
    frame = base64_to_cv2(img_bytes)
    cv2.imshow('res', frame)
    cv2.waitKey(0)






















