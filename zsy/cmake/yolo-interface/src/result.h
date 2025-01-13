#ifndef RESULT_H
#define RESULT_H

#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

// @brief 处理yolov5的结果
// @note __declspec(dllexport) 导出类识别标志，不加会出错
__declspec(dllexport) class ResultYolo {
public:
	//std::vector<std::string> class_names;
	float factor;

	//ResultYolov5();
	void read_class_names(std::string path_name);
	cv::Mat yolov5_result(cv::Mat image, float* result);
	cv::Mat yolov8_result(cv::Mat image, float* result);
	std::vector<std::string> class_names = { "left", "right" };
	//std::vector<std::string> class_names = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
	//std::vector<std::string> classes{"person"};

};


#endif // !RESULT_H
