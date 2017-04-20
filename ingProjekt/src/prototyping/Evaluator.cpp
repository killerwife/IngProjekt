#include "Evaluator.h"

using namespace CV;

Evaluator::Evaluator(std::vector<std::string> &cascadeXML)
{
    for (std::string& fileName : cascadeXML)
    {
        cv::Ptr<CV::BoostOwn> boost = cv::Algorithm::load<CV::BoostOwn>(fileName);
        m_cascade.push_back(boost);
    }
}


Evaluator::~Evaluator()
{
}

/*
Fills data matrices with the following:
data is filled with one row per picture, responses is filed with one response value per picture
backfitting specifies whether a secondary false positive folder should be used
countPos, countNeg and countBackfit specify amounts of each respective cathegory
sampleFolders[0] specifies folder for positive samples
sampleFolders[1] specifies folder for negative samples
sampleFolders[2] specifies folder for backfitting samples
*/
void Evaluator::fillData(cv::Mat& data, cv::Mat& responses, bool backfitting, int countPos, int countNeg, std::string sampleFolders[3], int countBackfit)
{
    Parser parser;
    data = cv::Mat(0, 0, CV_32S);
    responses = cv::Mat(0, 0, CV_32S);
    int arrayPos[1] = { 0 };
    cv::Mat pos(1, 1, CV_32S, arrayPos);
    parser.toMat(data, responses, sampleFolders[0], countPos, pos);
    int arrayNeg[1] = { 1 };
    cv::Mat neg(1, 1, CV_32S, arrayNeg);
    parser.toMat(data, responses, sampleFolders[1], countNeg, neg);
    if (backfitting)
        parser.toMat(data, responses, sampleFolders[2], countBackfit, neg);
}

/*
Fills data matrices with the following:
data is filled with all pictures, responses is filed with one response value per picture
backfitting specifies whether a secondary false positive folder should be used
countPos, countNeg and countBackfit specify amounts of each respective cathegory
sampleFolders[0] specifies folder for positive samples
sampleFolders[1] specifies folder for negative samples
sampleFolders[2] specifies folder for backfitting samples
*/
void Evaluator::fillData(std::vector<cv::Mat>& data, cv::Mat& responses, bool backfitting, int countPos, int countNeg, std::string sampleFolders[3], int countBackfit)
{
    Parser parser;
    responses = cv::Mat(0, 0, CV_32S);
    int arrayPos[1] = { 1 };
    cv::Mat pos(1, 1, CV_32S, arrayPos);
    parser.toMat(data, responses, sampleFolders[0], countPos, pos);
    int arrayNeg[1] = { 0 };
    cv::Mat neg(1, 1, CV_32S, arrayNeg);
    parser.toMat(data, responses, sampleFolders[1], countNeg, neg);
    if (backfitting)
        parser.toMat(data, responses, sampleFolders[2], countBackfit, neg);
}

/*
Trains an XML with sample data.
Backfitting specifies whether a secondary false negative folder should be used.
sampleFolders[0] specifies folder for positive samples
sampleFolders[1] specifies folder for negative samples
sampleFolders[2] specifies folder for backfitting samples
*/
void Evaluator::trainintRawImage(bool backfitting, std::string xml, std::string sampleFolders[3])
{
    cv::Mat data;
    cv::Mat responses;
    fillData(data, responses, backfitting, 5000, 20000, sampleFolders, 8000);
    cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();
    //cv::Ptr<cv::ml::TrainData> trainData = prepare_train_data(*data,*responses,40);
    //cv::FileStorage fs1("data.yml", cv::FileStorage::WRITE);
    //fs1 << "yourMat" << *data;
    //cv::FileStorage fs2("responses.yml", cv::FileStorage::WRITE);
    //fs2 << "yourMat" << *responses;
    boost->setBoostType(cv::ml::Boost::REAL);
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(1);
    boost->setUseSurrogates(false);
    boost->setCVFolds(0);
    boost->train(data, cv::ml::ROW_SAMPLE, responses); // 'prepare_train_data' returns an instance of ml::TrainData class
    boost->save(xml);
}

void Evaluator::detectRawImage(std::string filename, bool backfitting, std::string sampleFolders[3])
{
	Parser parser;
	cv::Mat data;
	cv::Mat responses;
	fillData(data, responses, backfitting, 40000, 40000, sampleFolders, 40000);
	cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(filename);
	std::vector< cv::Rect > faces;
	cv::Mat result;
	boost->predict(data, result);
	float *dataz = (float*)result.data;
	long *resultz = (long*)responses.data;
	//for (int i = 0; i < data->rows; i++)
	//{
	//	printf("Poradove cislo:%d Vysledok:%f Spravny Vysledok: %u\n", i, dataz[i], resultz[i]);
	//}
	int spravne = 0, nespravne = 0;
	for (int i = 0; i < data.rows; i++)
	{
		if (dataz[i] == resultz[i]) spravne++;
		else nespravne++;
	}
	printf("Spravnych vysledkov:%d Nespravnych:%d\n", spravne, nespravne);
}

/*
Trains an XML with sample data.
Backfitting specifies whether a secondary false negative folder should be used.
sampleFolders[0] specifies folder for positive samples
sampleFolders[1] specifies folder for negative samples
sampleFolders[2] specifies folder for backfitting samples
*/
void Evaluator::trainint(bool backfitting, std::string xml, std::string sampleFolders[3])
{
    std::vector<cv::Mat> data;
    cv::Mat responses;
    fillData(data, responses, backfitting, 500, 2000, sampleFolders, 200);
    //for (auto &mat : data)
    //{
    //    cv::resize(mat, mat, cv::Size(24, 40));
    //}

    cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();
	HaarTransform transform(data.size(),data[0].size());
	transform.SetImages(data, responses);
	cv::Mat trainingData;
	transform.GetFeatures(trainingData);
	long *dataz = (long*)responses.data;
	//for (int i = 0; i < 1500; i++)
	//{
	//	printf("%d:%d\t", i, dataz[i]);
	//}
	//cv::imshow("bla",data[0]);
	//cv::waitKey(0);
	//cv::imshow("bla", data[1400]);
	//cv::waitKey(0);
    //cv::Ptr<cv::ml::TrainData> trainData = prepare_train_data(*data,*responses,40);
    //cv::FileStorage fs1("data.yml", cv::FileStorage::WRITE);
    //fs1 << "yourMat" << *data;
    //cv::FileStorage fs2("responses.yml", cv::FileStorage::WRITE);
    //fs2 << "yourMat" << *responses;
    boost->setBoostType(cv::ml::Boost::REAL);
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.98);
    boost->setMaxDepth(1);
    boost->setUseSurrogates(false);
    boost->setCVFolds(0);
	boost->train(trainingData, cv::ml::ROW_SAMPLE, responses); // 'prepare_train_data' returns an instance of ml::TrainData class
    boost->save(xml);
}

void Evaluator::detect(bool backfitting, std::string filename, std::string sampleFolders[3])
{
    Parser parser;
	std::vector<cv::Mat> data;
    cv::Mat responses;
    fillData(data, responses, backfitting, 2000, 4000, sampleFolders, 1000);
    //for (auto &mat : data)
    //{
    //    cv::resize(mat, mat, cv::Size(24, 40));
    //}
    cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(filename);
	HaarTransform transform(data.size(), data[0].size());
	transform.SetImages(data, responses);
	cv::Mat trainingData;
	transform.GetFeatures(trainingData);
    std::vector< cv::Rect > faces;
    cv::Mat result;
	boost->predict(trainingData, result);
    float *resultz = (float*)result.data;
    //for (int i = 0; i < data->rows; i++)
    //{
    //	printf("Poradove cislo:%d Vysledok:%f Spravny Vysledok: %u\n", i, dataz[i], resultz[i]);
    //}
	//printf("%d\n",data.size());
	long *dataz = (long*)responses.data;
	//for (int i = 0; i < 200; i++)
	//{
	//	printf("%d:%d\t",i,dataz[i]);
	//}
	int positives, negatives, falsepositives, falsenegatives;
	positives = negatives = falsepositives = falsenegatives = 0;
    for (int i = 0; i < data.size(); i++)
    {
		if (dataz[i] == 1)
		{
			if (dataz[i] == resultz[i])
				positives++;
			else
				falsenegatives++;
		}
		else
		{
			if (dataz[i] == resultz[i])
				negatives++;
			else
				falsenegatives++;
		}
    }
    printf("Positives vysledkov:%d False positives:%d\n", positives, falsepositives);
	printf("Negatives vysledkov:%d False negatives:%d\n", negatives, falsenegatives);
}

/*
Multiscale detection using a trained boost model.
exportShit specifies whether found detection regions should be exported into images on the hard drive (used for backfitting)
filename is the path used for saving final detection result
imageName is the image upon which we want to launch the detection algorithm
*/
void Evaluator::detectMultiScale(bool exportShit, std::string xml, std::string filename, std::string imageName)
{
	cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(xml);
	cv::Mat img = cv::imread(imageName,CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat img = cv::imread("C:\\GitHubCode\\IngProjekt\\ingProjekt\\ingProjekt\\" + imageName);
	cv::Mat result = img.clone();
	std::vector<cv::Rect> boundingBoxes;
	HaarTransform transform(1,cv::Size(96,160));
	std::vector< cv::Rect > faces;
	//FILE * file = fopen("rects.txt", "w");
	int shift = 4;
	int m = 32000;
	for (int scale = 32; scale <= 512; scale += 4, shift += 4)
	{
		printf("%d\n", scale);
		for (int i = 0; i + scale * 3 <= img.cols; i += shift)
		{
			for (int k = 0; k + scale * 5 <= img.rows; k += shift)
			{
				cv::Rect rectangleZone(i, k, scale * 3, scale * 5);
				cv::Mat imagePart = cv::Mat(img, rectangleZone);
				//cv::resize(imagePart, imagePart, cv::Size(96, 160));
				//imagePart.convertTo(imagePart, CV_32F);
				//cv::Mat imagePartInput = imagePart.reshape(1, 1);
				transform.SetImage(imagePart, 0);
				cv::Mat trainingData;
				transform.GetFeatures(trainingData);
				cv::Mat response;
				boost->predict(trainingData, response);
				long* responses = (long*)response.data;
				if (responses[0] == 1)
				{
					//rectangle(result, rectangleZone, (0, 0, 255), 2);
					boundingBoxes.push_back(rectangleZone);
					//fprintf(file, "%d %d %d %d\n", rectangleZone.x, rectangleZone.y, rectangleZone.width, rectangleZone.height);
					if (exportShit)
					{
						cv::imwrite("C:\\GitHubCode\\backfitting\\pic" + std::to_string(m) + ".png", imagePart);
						m++;
					}

				}
			}
		}
	}
	//auto resultBoundingBoxes = nonMaxSuppression(boundingBoxes, 0.3f, 4);
	for (cv::Rect& box : boundingBoxes)
	{
		rectangle(result, box, (0, 0, 255), 2);
	}
	cv::imshow("Image", result);
	cv::waitKey(0);
	//fclose(file);
}

void Evaluator::detectMultiScaleTemp(bool exportShit, std::string xml, std::string filename, std::string imageName)
{
	cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(xml);
	cv::Mat img = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
	//cv::Mat img = cv::imread("C:\\GitHubCode\\IngProjekt\\ingProjekt\\ingProjekt\\" + imageName);
	cv::Mat resultImage = img.clone();
	cv::Mat responses;
	std::vector<cv::Rect> boundingBoxes;
	HaarTransform transform(1, cv::Size(96, 160));
	std::vector<cv::Mat> data;
	data.push_back(img);
	int arrayNeg[1] = { 1 };
	cv::Mat neg(1, 1, CV_32S, arrayNeg);
	responses.push_back(neg);
	//FILE * file = fopen("rects.txt", "w");
	//auto resultBoundingBoxes = nonMaxSuppression(boundingBoxes, 0.3f, 4);
	transform.SetImages(data, responses);
	cv::Mat trainingData;
	transform.GetFeatures(trainingData);
	cv::Mat result;
	boost->predict(trainingData, result);
	float *resultz = (float*)result.data;
	if (resultz[0]==1)
	rectangle(resultImage, cv::Rect(4,4,80,150), (0, 0, 255), 2);
	cv::imshow("Image", resultImage);
	//cv::imshow("Image", img);
	cv::waitKey(0);
	//fclose(file);
}


void Evaluator::detectMultiScaleProto(bool exportShit, std::string xml, std::string filename, std::string imageName)
{
    cv::Ptr<CV::BoostOwn> boost = cv::Algorithm::load<CV::BoostOwn>("..\\XML3.0\\" + xml);
    cv::Mat img = cv::imread("D:\\Anotacie\\03_Kajanek\\" + imageName);
    //cv::Mat img = cv::imread("C:\\GitHubCode\\IngProjekt\\ingProjekt\\ingProjekt\\" + imageName);
    cv::Mat result = img.clone();
    std::vector<cv::Rect> boundingBoxes;
    cv::Size scanningWindow(24, 40);
    HaarTransform transform(HaarFeatureParameters::ALL, scanningWindow);
    transform.SetImageBig(img);
    std::vector< cv::Rect > faces;
    //FILE * file = fopen("rects.txt", "w");
    int shift = 4;
    int m = 32000;
    int scaleCount = 0;
    long count = 0;
    cv::Mat features(1, transform.GetFeatureCount(), CV_32F);
    for (double scaleFactor = 1;; scaleFactor *= 1.1,scaleCount++)
    {
        printf("%lf\n", scaleFactor);
        cv::Size tempSize(std::round(img.cols*(1. / scaleFactor)), std::round(img.rows*(1. / scaleFactor)));
        if (scanningWindow.height > tempSize.height || scanningWindow.width > tempSize.width)
            break;

        cv::Size bounds(tempSize.width - scanningWindow.width, tempSize.height - scanningWindow.height);
        for (int i = 0; i < bounds.width; i += 2)
        {
            for (int k = 0; k < bounds.height; k += 2)
            {                
                cv::Rect rectangleZone(std::round(i*scaleFactor),std::round(k*scaleFactor),std::round(scanningWindow.width*scaleFactor),std::round(scanningWindow.height*scaleFactor));
                transform.CalculateFeatureVector(features,scaleCount,i,k);
                cv::Mat response;
                boost->predict(features, response);
                long* responses = (long*)response.data;
                if (responses[0] == 1)
                    boundingBoxes.push_back(rectangleZone);

                count++;
            }
        }
    }
    printf("Loop executed %d times.\n",count);
    //auto resultBoundingBoxes = nonMaxSuppression(boundingBoxes, 0.3f, 4);
    for (cv::Rect& box : boundingBoxes)
    {
        rectangle(result, box, (0, 0, 255), 2);
    }
    cv::imwrite(filename, result);
    cv::waitKey(0);
    //fclose(file);
}
