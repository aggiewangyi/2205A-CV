import onnxruntime  # 导入ONNX Runtime库，用于加载和执行ONNX模型
from tool import *  # 从tool模块导入所有内容（假设tool模块包含一些工具函数或类）
import ast  # 导入抽象语法树模块，用于处理Python源代码的抽象语法结构


# 定义函数onnx_load，用于加载ONNX模型
def onnx_load(w):
    cuda = torch.cuda.is_available()  # 检查CUDA是否可用
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else [
        'CPUExecutionProvider']  # 根据CUDA可用性设置执行提供者
    session = onnxruntime.InferenceSession(w, providers=providers)  # 创建ONNX Runtime推理会话
    output_names = [x.name for x in session.get_outputs()]  # 获取模型输出节点的名称
    return session, output_names  # 返回会话和输出节点名称


# 导入Flask和其他必要的库
from flask import Flask, request, render_template
import cv2 as cv  # 导入OpenCV库，用于图像处理

# 创建Flask应用实例
app = Flask(__name__, template_folder='.')  # 指定模板文件夹为当前目录

# 定义ONNX模型路径和图像尺寸
w = "yolov5l6.onnx"
imgsz = [640, 640]
# 加载ONNX模型
session, output_names = onnx_load(w)
# 从模型元数据中获取标签名称
label_name = session.get_modelmeta().custom_metadata_map['names']
label_name = ast.literal_eval(label_name)  # 将字符串形式的列表转换为真正的列表

# 设置设备（这里虽然设置了，但在后续代码中未使用，因为ONNX Runtime已根据CUDA可用性选择了执行提供者）
device = torch.device("cuda:0")

# 导入其他必要的库
import base64  # 用于图像的编码和解码
import numpy as np  # 用于数值计算


# 定义函数cv2_to_base64，用于将图像编码为base64字符串
def cv2_to_base64(image):
    image = cv2.imencode('.jpg', image)[1]  # 将图像编码为JPEG格式
    return base64.b64encode(image).decode('utf-8')  # 将编码后的图像转换为base64字符串


# 定义函数base64_to_cv2，用于将base64字符串解码为图像
def base64_to_cv2(base64_string):
    image_data = base64.b64decode(base64_string)  # 将base64字符串解码为二进制数据
    data = np.frombuffer(image_data, np.uint8)  # 将二进制数据转换为NumPy数组
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将NumPy数组解码为图像
    return image


# 定义Flask路由和处理函数
@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':  # 如果请求方法是POST
        file = request.files.get('file')  # 从请求中获取文件
        file = file.read()  # 读取文件内容
        file = np.frombuffer(file, dtype=np.uint8)  # 将文件内容转换为NumPy数组
        im0 = cv.imdecode(file, cv.IMREAD_COLOR)  # 将NumPy数组解码为图像
        # 对图像进行预处理
        im, org_data = data_process_cv2(im0, imgsz)  # 假设data_process_cv2是一个预处理函数，但代码中未定义
        # 使用ONNX模型进行推理
        y = session.run(output_names, {session.get_inputs()[0].name: im})  # 执行推理并获取输出
        pred = torch.from_numpy(y[0]).to(device)  # 将输出转换为Tensor并移动到设备（尽管这里设置了设备，但实际上是多余的）
        # 对推理结果进行非极大值抑制处理
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45,
                                   max_det=1000)  # 假设non_max_suppression是一个处理函数，但代码中未定义
        # 对处理后的结果进行后处理，得到图像和标签列表
        res_img, name_list = post_process_yolov5(pred[0], im0, label_name,
                                                 org_data)  # 假设post_process_yolov5是一个后处理函数，但代码中未定义
        name_list = str(name_list)  # 将标签列表转换为字符串（这里可能不是最佳做法，因为后续可能需要列表形式）
        res_img = cv2_to_base64(res_img)  # 将结果图像编码为base64字符串
        # 渲染模板并返回结果
        return render_template('./templates/home2.html', base64_str=f'{res_img}', product=f'{name_list}')
    return render_template('./templates/home2.html')  # 如果请求方法是GET，则直接渲染模板


# 设置Flask应用的配置
app.config['UPLOAD_FOLDER'] = './static/img'  # 设置上传文件夹（但代码中未使用上传功能）
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # 设置允许的文件扩展名（但代码中未使用文件扩展名检查）
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 设置最大内容长度

# 运行Flask应用
app.run(host='0.0.0.0', port=6008, debug=True)  # 在所有网络接口上运行应用，监听6008端口，并启用调试模式
