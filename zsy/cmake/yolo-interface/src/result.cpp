#include "result.h"
#include <windows.h>


void ResultYolo::read_class_names(std::string path_name)
{
	std::ifstream infile;
	infile.open(path_name.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	std::string str;
	while (getline(infile, str)) {
		class_names.push_back(str);
		str.clear();
	}
	infile.close();             //关闭文件输入流 

}

cv::Mat ResultYolo::yolov5_result(cv::Mat image, float* result) {
	cv::Mat det_output = cv::Mat(15120, 85, CV_32F, result);
	//// post-process
	std::vector<cv::Rect> position_boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;

	float thresh = 0.1;
	std::cout << det_output.rows << std::endl;
	for (int i = 0; i < det_output.rows; i++) {
		float confidence = det_output.at<float>(i, 4);
		if (confidence < thresh) {
			continue;
		}
		std::cout << "confidence: " << confidence << std::endl;
		cv::Mat classes_scores = det_output.row(i).colRange(5, 85);
		cv::Point classIdPoint;
		double score;
		// 获取一组数据中最大值及其位置
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > thresh)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * factor);
			int y = static_cast<int>((cy - 0.5 * oh) * factor);
			int width = static_cast<int>(ow * factor);
			int height = static_cast<int>(oh * factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			position_boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}
	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(position_boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		cv::rectangle(image, position_boxes[index], cv::Scalar(0, 0, 255), 2, 8);
		cv::rectangle(image, cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 20),
			cv::Point(position_boxes[index].br().x, position_boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		cv::putText(image, class_names[idx], cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));		
		//查看输出结果
		//cv::imshow("C++ + OpenVINO + Yolov5 推理结果", image);
		//cv::waitKey();
	}
	return image;
}


cv::Mat ResultYolo::yolov8_result(cv::Mat image, float* result) {
	cv::Mat det_output = cv::Mat(6, 8400, CV_32F, result);
	//std::vector<cv::Mat> outputs;
	int rows = det_output.size[1];
	int dimensions = det_output.size[0];

	det_output = det_output.reshape(1, dimensions);
	cv::transpose(det_output, det_output);
	
	float* data = (float*)det_output.data;
	float* classes_scores = data + 4;
	//// post-process
	std::vector<cv::Rect> position_boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;

	float thresh = 0.5;
	//std::cout << det_output.rows << std::endl;
	//DWORD start_time, end_time;

	//start_time = timeGetTime();
	for (int i = 0; i < det_output.rows; i++) {
		
		float* scores_point = data + 4;

		cv::Mat classes_scores(1, class_names.size(), CV_32FC1, scores_point);

		cv::Point classIdPoint;
		double score;
		// 获取一组数据中最大值及其位置
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 置信度 0～1之间
		if (score > thresh)
		{
			float cx = data[0];
			float cy = data[1];
			float ow = data[2];
			float oh = data[3];
			int x = static_cast<int>((cx - 0.5 * ow) * factor);
			int y = static_cast<int>((cy - 0.5 * oh) * factor);
			int width = static_cast<int>(ow * factor);
			int height = static_cast<int>(oh * factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			position_boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
		data += dimensions;
	}
	//end_time = timeGetTime();
	//printf("Post Time 0:%f \n", (end_time - start_time) * 1.0);

	// NMS
	std::vector<int> indexes;
	//start_time = timeGetTime();
	cv::dnn::NMSBoxes(position_boxes, confidences, 0.25, 0.45, indexes);
	//end_time = timeGetTime();
	//printf("Post Time 1:%f \n", (end_time - start_time) * 1.0);

	//start_time = timeGetTime();
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		cv::rectangle(image, position_boxes[index], cv::Scalar(0, 0, 255), 2, 8);
		cv::rectangle(image, cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 20),
			cv::Point(position_boxes[index].br().x, position_boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		cv::putText(image, class_names[idx], cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		//查看输出结果
		/*cv::imshow("C++ + OpenVINO + Yolov5 推理结果", image);
		cv::waitKey();*/
	}
	//end_time = timeGetTime();
	//printf("Post Time 2:%f \n", (end_time - start_time) * 1.0);


	return image;
}