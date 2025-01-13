#include <windows.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <time.h>
#include <opencv2/opencv.hpp>

#include "dirent.h"
#include "result.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#pragma comment(lib,"winmm.lib")


// @brief 用于创建IBuilder、IRuntime或IRefitter实例的记录器用于通过该接口创建的所有对象。
// 在释放所有创建的对象之前，记录器应一直有效。
// 主要是实例化ILogger类下的log()方法。
class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* message)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << message << std::endl;
	}
} gLogger;



void onnx_to_engine(std::string onnx_file_path, std::string engine_file_path, int type) {

	// 构建器，获取cuda内核目录以获取最快的实现
	// 用于创建config、network、engine的其他对象的核心类
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// 解析onnx网络文件
	// tensorRT模型类
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx文件解析类
	// 将onnx文件解析，并填充rensorRT网络结构
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	// 解析onnx文件
	parser->parseFromFile(onnx_file_path.c_str(), 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// 创建推理引擎
	// 创建生成器配置对象。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 设置最大工作空间大小。
	config->setMaxWorkspaceSize(16 * (1 << 20));
	// 设置模型输出精度
	if (type == 1) {
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	if (type == 2) {
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
	}
	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// 将推理银枪保存到本地
	std::cout << "try to save engine file now !" << std::endl;
	std::ofstream file_ptr(engine_file_path, std::ios::binary);
	if (!file_ptr) {
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	// 将模型转化为文件流数据
	nvinfer1::IHostMemory* model_stream = engine->serialize();
	// 将文件保存到本地
	file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
	// 销毁创建的对象
	model_stream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}


void getAllFiles(const char* path, std::vector<std::string>& files)
{
	DIR* dir = opendir(path);
	struct dirent* entry;

	if (dir != NULL){ 
		while ((entry = readdir(dir)) != NULL) {			
			if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
				continue;
			//std::cout << entry->d_name << std::endl; // 输出每个文件名
			files.push_back(entry->d_name);
		}
	}
	else {
		perror("无法打开目录");
		return;
	}
	return;
}


int main() {

	
   //当目录中文件数量多时读取和推理速度会明显下降
	//const char* image_dir = "E:/coco/images/train2017";
	const char* image_dir = "E:/fish/picture";
	std::vector<std::string> files_list;
	getAllFiles(image_dir, files_list);

	const char* result_dir = "F:/CPP-Project/trt_inference/result_dir";

	//const char* model_path_onnx = "F:/CPP-Project/Inference/model/yolov5/yolov5s.onnx";
	//const char* model_path_engine = "F:/CPP-Project/Inference/model/yolov5/yolov5s.engine";
	//const char* model_path_onnx = "F:/CPP-Project/trt_inference/best-person.onnx";

	//const char* model_path_onnx = "E:/PycharmProject/ultralytics/yolov8n.onnx";
	//const char* model_path_engine = "F:/CPP-Project/trt_inference/best-person.engine";
   // engine文件必须在自己电脑上用onnx转换，不能直接用其他电脑上复制过来的
	//const char* model_path_engine = "F:/CPP-Project/trt_inference/yolov8n.engine";
	const char* model_path_engine = "F:/CPP-Project/trt_inference/fish_best_int8.engine";
	//const char* image_path = "F:/CPP-Project/Inference/model/yolov5/text_image/0001.jpg";
	//std::string lable_path = "F:/CPP-Project/Inference/model/yolov5/lable.txt";
	const char* input_node_name = "images";
	const char* output_node_name = "output0";
	int num_ionode = 2;

	/*std::string model_path_onnx = "F:/CPP-Project/trt_inference/fish_best.onnx";
	std::string engine_file_path = "F:/CPP-Project/trt_inference/fish_best_int8.engine";
	int type = 2;
	onnx_to_engine(model_path_onnx, engine_file_path, type);
	return 0;*/

	// 读取本地模型文件
	std::ifstream file_ptr(model_path_engine, std::ios::binary | std::ios::in);
	if (!file_ptr.good()) {
		std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
	}

	size_t size = 0;
	file_ptr.seekg(0, file_ptr.end);	// 将读指针从文件末尾开始移动0个字节
	size = file_ptr.tellg();			// 返回读指针的位置，此时读指针的位置就是文件的字节数
	file_ptr.seekg(0, file_ptr.beg);	// 将读指针从文件开头开始移动0个字节
	char* model_stream = new char[size];
	file_ptr.read(model_stream, size);
	file_ptr.close();
	
	// 日志记录接口
	Logger logger;
	// 反序列化引擎
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	// 推理引擎
	// 保存模型的模型结构、模型参数以及最优计算kernel配置；
	// 不能跨平台和跨TensorRT版本移植
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_stream, size);
	// 上下文
	// 储存中间值，实际进行推理的对象
	// 由engine创建，可创建多个对象，进行多推理任务
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	

	delete[] model_stream;
	// 创建GPU显存缓冲区
	void** data_buffer = new void* [num_ionode];
	// 创建GPU显存输入缓冲区
	int input_node_index = engine->getBindingIndex(input_node_name);
	nvinfer1::Dims input_node_dim = engine->getBindingDimensions(input_node_index);
	size_t input_data_length = input_node_dim.d[1]* input_node_dim.d[2] * input_node_dim.d[3];
	cudaMalloc(&(data_buffer[input_node_index]), input_data_length * sizeof(float));
	// 创建GPU显存输出缓冲区
	int output_node_index = engine->getBindingIndex(output_node_name);
	nvinfer1::Dims output_node_dim = engine->getBindingDimensions(output_node_index);
	size_t output_data_length = output_node_dim.d[1] * output_node_dim.d[2] ;
	cudaMalloc(&(data_buffer[output_node_index]), output_data_length * sizeof(float));
	cv::Size input_node_shape(input_node_dim.d[2], input_node_dim.d[3]);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	std::vector<float> input_data(input_data_length);
	float* result_array = new float[output_data_length];
	
	//cv::Mat image;
	ResultYolo result;
	cv::Mat result_image;
	//cv::Mat BN_image;
	//cv::Mat max_image;
	DWORD start_time, end_time;

	//result.read_class_names(lable_path);

	for(int index=0; index< files_list.size(); index++){
		//start_time = timeGetTime();

		std::string image_path = (std::string)image_dir +"/"+files_list[index];

		std::string save_path = (std::string)result_dir + "/" + files_list[index];
		
		//std::cout << "image path: " << image_path << std::endl;
		// 图象预处理 - 格式化操作
		//start_time = timeGetTime();
		cv::Mat image = cv::imread(image_path);
		//end_time = timeGetTime();
		//printf("Inference Time 0:%f \n", (end_time - start_time) * 1.0);

		start_time = timeGetTime();
		int max_side_length = std::max(image.cols, image.rows);
		cv::Mat max_image = cv::Mat::zeros(cv::Size(max_side_length, max_side_length), CV_8UC3);
		/*end_time = timeGetTime();
		printf("Inference Time 1:%f \n", (end_time - start_time) * 1.0);*/

		//start_time = timeGetTime();
		cv::Rect roi(0, 0, image.cols, image.rows);
		image.copyTo(max_image(roi));
		/*end_time = timeGetTime();
		printf("Inference Time 2:%f \n", (end_time - start_time) * 1.0);*/


		//start_time = timeGetTime();
		// 将图像归一化，并放缩到指定大小	
		cv::Mat BN_image = cv::dnn::blobFromImage(max_image, 1 / 255.0, input_node_shape, cv::Scalar(0, 0, 0), true, false);
		/*end_time = timeGetTime();
		printf("Inference Time 3:%f \n", (end_time - start_time) * 1.0);*/

		// 创建输入cuda流
	/*	cudaStream_t stream;
		cudaStreamCreate(&stream);*/
		
		//start_time = timeGetTime();
		memcpy(input_data.data(), BN_image.ptr<float>(), input_data_length * sizeof(float));
		// 输入数据由内存到GPU显存
		cudaMemcpyAsync(data_buffer[input_node_index], input_data.data(), input_data_length * sizeof(float), cudaMemcpyHostToDevice, stream);
		// 模型推理
		context->enqueueV2(data_buffer, stream, nullptr);
		cudaMemcpyAsync(result_array, data_buffer[output_node_index], output_data_length * sizeof(float), cudaMemcpyDeviceToHost, stream);

		result.factor = max_side_length / (float) input_node_dim.d[2];	
		/*end_time = timeGetTime();
		printf("Inference Time 4:%f\n", (end_time - start_time) * 1.0);*/
		//cv::Mat result_image = result.yolov5_result(image, result_array);		
		
		//start_time = timeGetTime();
		result_image = result.yolov8_result(image, result_array);
		end_time = timeGetTime();
		printf("Post Process Time:%f \n\n", (end_time - start_time)*1.0);

		 cv::imwrite(save_path, result_image);
		//printf("Spend Time:%f \n", (end_time - start_time)*1.0);
		// 查看输出结果
		/*cv::imshow("C++ trt Yolo 推理结果", result_image);
		cv::waitKey();*/
	}
}
