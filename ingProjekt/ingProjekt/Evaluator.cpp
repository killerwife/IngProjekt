#include "Evaluator.h"


Evaluator::Evaluator()
{
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

/*
Trains an XML with sample data.
Backfitting specifies whether a secondary false negative folder should be used.
sampleFolders[0] specifies folder for positive samples
sampleFolders[1] specifies folder for negative samples
sampleFolders[2] specifies folder for backfitting samples
*/
void Evaluator::trainint(bool backfitting, std::string xml, std::string sampleFolders[3])
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

void Evaluator::detect(std::string filename, bool backfitting, std::string sampleFolders[3])
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
