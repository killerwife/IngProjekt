#include "Evaluator.h"
#include <ctime>
#include "opencv2\cudaobjdetect.hpp"

using namespace CV;

void detection(std::string model, std::string file)
{
	//for time measure  
	double TakeTime;
	unsigned long long Atime, Btime;

	//window  
	//cv::namedWindow("origin");

	//load image  
	cv::Mat img = cv::imread(file);
	cv::Mat grayImg; //adaboost detection is gray input only.  
	cvtColor(img, grayImg, CV_BGR2GRAY);

	//load xml file  
	std::string trainface = model;

	//declaration  
	cv::CascadeClassifier ada_cpu;
    cv::Ptr<cv::cuda::CascadeClassifier> ada_gpu = cv::cuda::CascadeClassifier::create(model);

	if (!(ada_cpu.load(trainface)))
	{
		printf(" cpu ada xml load fail! \n");
		return;
	}

	//if (!(ada_gpu.load(trainface)))
	//{
	//    printf(" gpu ada xml load fail! \n");
	//    return;
	//}

	//////////////////////////////////////////////  
	//cpu case face detection code  
	std::vector< cv::Rect > faces;
	//Atime = cv::getTickCount();
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
	ada_cpu.detectMultiScale(grayImg, faces);
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "CPU finished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";
	//Btime = cv::getTickCount();
	//TakeTime = (Btime - Atime) / cv::getTickFrequency();
	//printf("detected face(cpu version) = %d / %lf sec take.\n", faces.size(), TakeTime);
    if (faces.size() >= 1)
    {
        for (int ji = 0; ji < faces.size(); ++ji)
        {
            rectangle(img, faces[ji], CV_RGB(0, 0, 255), 4);
        }
    }

	/////////////////////////////////////////////  
	//gpu case face detection code  
	cv::cuda::GpuMat faceBuf_gpu;
	cv::cuda::GpuMat GpuImg;

    GpuImg.upload(grayImg);
    start = std::chrono::system_clock::now();
    ada_gpu->detectMultiScale(GpuImg, faceBuf_gpu);
    end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::chrono::system_clock::to_time_t(end);

    std::cout << "GPU finished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";
	//printf("detected face(gpu version) =%d / %lf sec take.\n", detectionNumber, TakeTime);

//    std::vector<cv::Rect> faces;
    ada_gpu->convert(faceBuf_gpu, faces);

    for (int i = 0; i < faces.size(); ++i)
        cv::rectangle(img, faces[i], cv::Scalar(255));


	/////////////////////////////////////////////////  
	//result display  
	imshow("origin", img);
	cv::waitKey(0);
}

static cv::Ptr<cv::ml::TrainData>
prepare_train_data(const cv::Mat& data, const cv::Mat& responses, int ntrain_samples)
{
	cv::Mat sample_idx = cv::Mat::zeros(1, data.rows, CV_8U);
	cv::Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(cv::Scalar::all(1));

	int nvars = data.cols;
	cv::Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(cv::Scalar::all(cv::ml::VAR_ORDERED));
	var_type.at<uchar>(nvars) = cv::ml::VAR_CATEGORICAL;

	return cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, responses,
		cv::noArray(), sample_idx, cv::noArray(), var_type);
}

/*
Comparator for sorting cv::Rect according to our needs
Currently unused
*/
struct RectComparator
{
	bool operator()(cv::Rect a, cv::Rect b)
	{
		return (a.y + a.height) < (b.y + b.height);
	}
};

/*
Performs flavor of non-maximum-suppression upon boundingBoxes, which contains all detected rectangles.
Overlap threshold is usually set to between 0.3-0.5.
Multiplier specifies the maximum area difference two bounding boxes can have to still be joined together
Multiplier 4 generally means a bounding box 2x the size
*/
std::vector<cv::Rect> nonMaxSuppression(std::vector<cv::Rect> boundingBoxes, float overlapThreshold, int multiplier)
{
	std::vector<cv::Rect> result;
	//std::sort(boundingBoxes.begin(), boundingBoxes.end(), RectComparator());
	/*for (cv::Rect rect : boundingBoxes)
		std::cout << rect << "\t";*/
	std::vector<double> areas;
	std::vector<int> indexes;
	std::vector<int> x;
	std::vector<int> y;
	std::vector<int> widths;
	std::vector<int> heights;
	std::vector<int> weights;
	int i = 0;
	for (cv::Rect box : boundingBoxes)
	{
		indexes.push_back(i++);
		x.push_back(box.x);
		y.push_back(box.y);
		widths.push_back(box.width);
		heights.push_back(box.height);
		areas.push_back((box.height + 1)*(box.width + 1));
		weights.push_back(1);
	}
	//while (indexes.size() > 0)
	//{
	//	int last = indexes.size() - 1;
	//	auto i = indexes[last];
	//	result.push_back(boundingBoxes[i]);
	//	std::vector<int> supress;
	//	supress.push_back(indexes[last]);
	//	for (int j : indexes)
	//	{
	//		int xx1 = x[i] > x[j] ? x[i] : x[j];
	//		int yy1 = y[i] > y[j] ? y[i] : y[j];
	//		int xx2 = x[i] + width[i] < x[j] + width[j] ? x[i] + width[i] : x[j] + width[j];
	//		int yy2 = y[i] + height[i] < x[j] + height[j] ? x[i] + height[i] : x[j] + height[j];
	//		float w = xx2 - xx1 + 1 > 0 ? xx2 - xx1 + 1 : 0;
	//		float h = yy2 - yy1 + 1 > 0 ? yy2 - yy1 + 1 : 0;
	//		float overlap = w*h / areas[j];
	//		if (overlap > overlapThreshold)
	//			supress.push_back(j);
	//	}
	//	for (int removalIdx : supress)
	//		indexes.erase(std::remove(indexes.begin(), indexes.end(), removalIdx), indexes.end());
	//}
	//std::vector<int> candidates;
	for (int i = 0; i < boundingBoxes.size();)
	{
		int candidate = -1;
		for (int j = 0; j < boundingBoxes.size(); j++)
		{
			if (i == j || (x[i]>x[j] + widths[j] || x[i] + widths[i]<x[j]) || (y[i]>y[j] + heights[j] || y[i] + heights[i] < y[j]))
				continue;

			int xx1 = x[i] > x[j] ? x[i] : x[j];
			int yy1 = y[i] > y[j] ? y[i] : y[j];
			int xx2 = x[i] + widths[i] < x[j] + widths[j] ? x[i] + widths[i] : x[j] + widths[j];
			int yy2 = y[i] + heights[i] < x[j] + heights[j] ? x[i] + heights[i] : x[j] + heights[j];
			double w = xx2 - xx1 + 1 > 0 ? xx2 - xx1 + 1 : 0;
			double h = yy2 - yy1 + 1 > 0 ? yy2 - yy1 + 1 : 0;
			double overlap = w*h / (areas[j] + areas[i]);
			double areaDiff = areas[i] / areas[j];
			if (overlap > overlapThreshold && (areaDiff > 1.f / multiplier && areaDiff < multiplier))
			{
				//candidates.push_back(j);
				candidate = j;
				break;
			}
			//printf("%d\t", j);
		}
		//printf("\n%d\n", i);
		if (candidate != -1) // TODO: add joining of several bounding boxes at once
		{
			int height = (heights[i] + heights[candidate]) / 2;
			int width = (widths[i] + widths[candidate]) / 2;
			int finalWeight = weights[candidate] + weights[i];
			int middleFirstX = x[i] + widths[i] / 2;
			int middleSecondX = ((x[candidate] + widths[candidate]) / 2);
			int middleFirstY = (y[i] + heights[i]) / 2;
			int middleSecondY = ((y[candidate] + heights[candidate]) / 2);
			int middleX = (int)((double)abs(middleFirstX - middleSecondX) / finalWeight*weights[candidate]);
			int middleY = (int)((double)abs(middleFirstY - middleSecondY) / finalWeight*weights[candidate]);
			int coordX = middleX - width / 2 + x[i] > x[candidate] ? x[candidate] : x[i];
			int coordY = middleY - height / 2 + y[i] > y[candidate] ? y[candidate] : y[i];
			boundingBoxes.push_back(cv::Rect(coordX, coordY, width, height));
			areas.push_back((height + 1)*(width + 1));
			x.push_back(coordX);
			y.push_back(coordY);
			widths.push_back(width);
			heights.push_back(height);
			weights.push_back(finalWeight);
			boundingBoxes.erase(boundingBoxes.begin() + i);
			areas.erase(areas.begin() + i);
			x.erase(x.begin() + i);
			y.erase(y.begin() + i);
			widths.erase(widths.begin() + i);
			heights.erase(heights.begin() + i);
			weights.erase(weights.begin() + i);
			if (i < candidate)
				candidate--;
			boundingBoxes.erase(boundingBoxes.begin() + candidate);
			areas.erase(areas.begin() + candidate);
			x.erase(x.begin() + candidate);
			y.erase(y.begin() + candidate);
			widths.erase(widths.begin() + candidate);
			heights.erase(heights.begin() + candidate);
			weights.erase(weights.begin() + candidate);
			//printf("Joined %d and %d\n", i, candidate);
			i = 0;
		}
		else
			i++;
	}
	return boundingBoxes;
}

/*
Function for non-maximum-suppression testing.
Enables quick loading of rectangles from file instead of requiring a detection algorithm run.
*/
void rectOnly(std::string imageName)
{
	FILE * file = fopen("rects.txt", "r");
	std::vector<cv::Rect> rects;
	cv::Rect temp;
	while (fscanf(file, "%d%d%d%d", &temp.x, &temp.y, &temp.width, &temp.height) != EOF)
		rects.push_back(temp);
	cv::Mat result = cv::imread("C:\\GitHubCode\\anotovanie\\" + imageName);
	//cv::imshow("bla", result);
	//cv::waitKey(0);
	auto resultBoundingBoxes = nonMaxSuppression(rects, 0.3f, 4);
	for (cv::Rect& box : resultBoundingBoxes)
	{
		rectangle(result, box, (0, 0, 255), 2);
	}
	cv::imwrite("trieska2.png", result);
	cv::waitKey(0);
	fclose(file);
}

/*
ocasovat cpu boost a gpu boost, trening a detekcia niekolkych obrazkov
*/

void commands(int defaultCommand = -1)
{
	if (!(defaultCommand == -1))
		scanf("%d", &defaultCommand);

	while (defaultCommand)
	{
		printf("Vyberte moznost:\n");
		printf("0: Koniec.\n");
		printf("1: Test obrazku na natrenovanom modeli pre detekciu tvari.\n");
		printf("2: Test obrazku na natrenovanom modeli pre detekciu tiel.\n");
		printf("3: Trening modelu na pribalenych datach.\n");
		printf("4: Benchmark natrenovaneho modelu.\n");
		printf("5: Test natrenovaneho modelu na jednom obrazku.\n");
		scanf("%d", &defaultCommand);
		switch (defaultCommand)
		{
		case 1:
			detection("haarcascade_frontalface_alt.xml", "happypeople.jpg");
			break;
		case 2:
			detection("haarcascade_fullbody.xml", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png");
			break;
		case 3:
		{
			std::string sampleFolders[3];
			sampleFolders[0] = "trenovacieData\\pos\\";
			sampleFolders[1] = "trenovacieData\\neg\\";
			sampleFolders[2] = "backfitting\\";
            std::vector<std::string> filenames;
            Evaluator eval(filenames);
			eval.trainint(false, "HaarXMLSemestralnaPraca.xml", sampleFolders);
			break;
		}
		case 4:
		{
			std::string sampleFolders[3];
			sampleFolders[0] = "testovacieData\\pos\\";
			sampleFolders[1] = "testovacieData\\neg\\";
			sampleFolders[2] = "backfitting\\";
            std::vector<std::string> filenames;
            Evaluator eval(filenames);
			eval.detect(false, "HaarXMLSemestralnaPraca.xml", sampleFolders);
			break;
		}
		case 5:
		{
			std::string pic = "20160428114934750.Png";
			printf("Zadajte meno obrazku: Default: 20160428114934750.Png\n");
			std::string line;
			std::getline(std::cin, line);
			std::getline(std::cin, line);
			if (line[0] != '\n' && line[0] != '\0')
				pic = line;
            std::vector<std::string> filenames;
            Evaluator eval(filenames);
			eval.detectMultiScaleTemp(false, "HaarXMLSemestralnaPraca.xml", "", pic);
			break;
		}
		case 0:
			break;
		}
	}
}

int main(int argc, char* argv[])
{
	/*std::thread thread1 = std::thread(trainint, false, "trainedBoostNoBackfit.xml");
	trainint(true, "trainedBoost.xml");
	thread1.join();*/
	//std::string sampleFolders[3];
	//sampleFolders[0] = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData\\";
	//sampleFolders[1] = "C:\\GitHubCode\\anotovanie\\TrainingData\\";
	//sampleFolders[2] = "C:\\GitHubCode\\backfitting\\";
	//trainint(true, "trainedBoostFinal3.xml",sampleFolders);
	//detect(true);
	/*std::thread thread1 = std::thread(detectMultiScale, false, "trainedBoost.xml", "outputBackfit.png");
	detectMultiScale(false, "trainedBoostNoBackfit.xml", "outputNoBackfit.png");
	thread1.join();*/

	//sampleFolders[0] = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Testing\\hrac\\RealData\\";
	//sampleFolders[1] = "C:\\GitHubCode\\anotovanie\\TrainingData\\";
	//sampleFolders[2] = "C:\\GitHubCode\\backfitting\\";

	//rectOnly("SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002000.png");
	//detectMultiScale(true, "trainedBoostFinal2.xml", "outputFinal2.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_001975.png");
	//detectMultiScale(false, "trainedBoostFinal0.xml", "outputFinalNotBackfitted1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002000.png");
	//detectMultiScale(false, "trainedBoostFinal3.xml", "outputFinalBackfitted1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002000.png");
	//trainint();
	//Parser parser;
	//parser.parseNegatives();
	//parser.parsePositives();

	//sampleFolders[0] = "D:\\Nenapadny priecinok\\neg\\";
	//sampleFolders[1] = "D:\\Nenapadny priecinok\\pos\\";
	/*sampleFolders[0] = "D:\\Nenapadny priecinok\\testData\\neg\\";
	sampleFolders[1] = "D:\\Nenapadny priecinok\\testData\\pos\\";*/
	//sampleFolders[2] = "C:\\GitHubCode\\backfitting\\";
	//   clock_t begin = clock();
    std::vector<std::string> filenames;
	Evaluator eval(filenames);
	//eval.trainint(false,"HaarXMLPrezentacia.xml",sampleFolders);
	//eval.detect(false, "HaarXMLPrezentacia.xml",sampleFolders);
	//eval.detectMultiScaleProto(false, "HaarXML2640.xml", "outputHaar1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png");
	//clock_t end = clock();
	//double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	//printf("Seconds %lf\n", elapsed_secs);
	//commands();

    //std::chrono::time_point<std::chrono::system_clock> start, end;
    //start = std::chrono::system_clock::now();
    //eval.detectMultiScaleProto(false, "HaarXML2640.xml", "outputHaar1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png");
    detection("haarcascade_fullbody.xml", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png");
    //end = std::chrono::system_clock::now();

    //std::chrono::duration<double> elapsed_seconds = end - start;
    //std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    //std::cout << "finished computation at " << std::ctime(&end_time)
    //    << "elapsed time: " << elapsed_seconds.count() << "s\n";
	return 0;
}