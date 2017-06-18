#include "Evaluator.h"
#include <ctime>
#include "opencv2\cudaobjdetect.hpp"
#include "FeaturePrototypes.h"
#include <fstream>

using namespace CV;

void detection(std::string model, std::string file, bool GPU, bool save = false)
{
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

    if (GPU)
    {
        /////////////////////////////////////////////  
        //gpu case face detection code  
        cv::Ptr<cv::cuda::CascadeClassifier> ada_gpu = cv::cuda::CascadeClassifier::create(model);

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
    }


    /////////////////////////////////////////////////  
    //result display  
    if (save)
        imwrite("..\\outputImages\\Output.png", img);
    else
    {
        cv::imshow("Output", img);
        cv::waitKey(0);
    }
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
            if (i == j || (x[i] > x[j] + widths[j] || x[i] + widths[i] < x[j]) || (y[i] > y[j] + heights[j] || y[i] + heights[i] < y[j]))
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

#define PATH_MAX 512

struct ObjectPos
{
    float x;
    float y;
    float width;
    bool found;    /* for reference */
};

void cascadePerformance(int argc, char* argv[])
{
    std::string classifierName;
    char* inputName = (char*)"";
    float maxSizeDiff = 0.5F;
    float maxPosDiff = 0.3F;
    double scaleFactor = 1.25;
    int minNeighbors = 3;
    cv::Size minSize;
    cv::Size maxSize;
    bool equalize = false;;
    bool saveDetected = false;
    FILE* info;
    cv::CascadeClassifier cascade;
    double totaltime = 0.0;

    if (argc == 2)
    {
        std::cout << "Aplikacia je urcena na vyhodnotenie vykonosti natrenovaneho detektora" << std::endl;
        std::cout << "Pouzitie: " << std::endl;
        std::cout << "  -classifier <classifier_directory_name>" << std::endl;
        std::cout << "  -input <collection_file_name>" << std::endl;
        std::cout << "  [-maxSizeDiff <max_size_difference = " << maxSizeDiff << ">]" << std::endl;
        std::cout << "  [-maxPosDiff <max_position_difference = " << maxPosDiff << ">]" << std::endl;
        std::cout << "  [-sf <scale_factor = " << scaleFactor << ">]" << std::endl;
        std::cout << "  [-minNeighbors <min_number_neighbors_for_each_candidate = " << minNeighbors << " >]" << std::endl;
        std::cout << "  [-minSize <min_possible_object_size> Example: 32x32 (Width * Height)]" << std::endl;
        std::cout << "  [-maxSize <max_possible_object_size> Example: 64x64 (Width * Height)]" << std::endl;
        std::cout << "  [-equalize <histogram_equalization: " << (equalize ? "True" : "False") << ">]" << std::endl; // ??
        std::cout << "  [-save <save_detection: " << (saveDetected ? "True" : "False") << ">]" << std::endl;
        return;
    }

    for (int i = 2; i < argc; i++)
    {
        if (!strcmp(argv[i], "-classifier"))
        {
            classifierName = argv[++i];
        }
        else if (!strcmp(argv[i], "-input"))
        {
            inputName = argv[++i];
        }
        else if (!strcmp(argv[i], "-maxSizeDiff"))
        {
            float tmp = (float)atof(argv[++i]);
            if (tmp >= 0 && tmp <= 1)
                maxSizeDiff = tmp;
        }
        else if (!strcmp(argv[i], "-maxPosDiff"))
        {
            float tmp = (float)atof(argv[++i]);
            if (tmp >= 0 && tmp <= 1)
                maxPosDiff = tmp;
        }
        else if (!strcmp(argv[i], "-sf"))
        {
            double tmp = atof(argv[++i]);
            if (tmp > 1)
                scaleFactor = tmp;
        }
        else if (!strcmp(argv[i], "-minNeighbors"))
        {
            minNeighbors = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-minSize"))
        {
            sscanf(argv[++i], "%ux%u", &minSize.width, &minSize.height);
        }
        else if (!strcmp(argv[i], "-maxSize"))
        {
            sscanf(argv[++i], "%ux%u", &maxSize.width, &maxSize.height);
        }
        else if (!strcmp(argv[i], "-equalize"))
        {
            equalize = true;
        }
        else if (!strcmp(argv[i], "-save"))
        {
            saveDetected = true;
        }
        else
            std::cerr << "WARNING: Neznama volba " << argv[i] << std::endl;
    }

    if (!cascade.load(classifierName))
    {
        std::cerr << "ERROR: Nemozem nacitat klasifikator" << std::endl;
        return;
    }

    char fullname[PATH_MAX];
    char detfilename[PATH_MAX];
    char* filename;
    char detname[] = "det-";

    // skopiruje do fullname cestu
    strcpy(fullname, inputName);
    // do filename vlozi smernik na posledny vyskyt '\\'
    filename = strrchr(fullname, '\\');
    if (filename == NULL)
    {
        // do filename vlozi smernik na posledny vyskyt znaku '/'
        filename = strrchr(fullname, '/');
    }
    if (filename == NULL)
    {
        filename = fullname;
    }
    else
    {
        filename++;
    }

    info = fopen(inputName, "r");
    if (info == NULL)
    {
        std::cerr << "ERROR: Nemozem otvorit vstupny subor" << std::endl;
        return;
    }

    std::cout << "Parametre: " << std::endl;
    std::cout << "Classifier: " << classifierName << std::endl;
    std::cout << "Input: " << inputName << std::endl;
    std::cout << "maxSizeDiff: " << maxSizeDiff << std::endl;
    std::cout << "maxPosDiff: " << maxPosDiff << std::endl;
    std::cout << "sf: " << scaleFactor << std::endl;
    std::cout << "minNeighbors: " << minNeighbors << std::endl;
    std::cout << "minSize: " << minSize << std::endl;
    std::cout << "maxSize: " << maxSize << std::endl;
    std::cout << "equalize: " << (equalize ? "True" : "False") << std::endl;
    std::cout << "save: " << (saveDetected ? "True" : "False") << std::endl;

    cv::Mat image, grayImage;
    int hits, missed, falseAlarms;
    int totalHits = 0, totalMissed = 0, totalFalseAlarms = 0, totalObjects = 0;
    int found;
    int refcount;

    std::cout << "+================================+======+======+======+=======+" << std::endl;
    std::cout << "|            File Name           | Hits |Missed| False|Objects|" << std::endl;
    std::cout << "+================================+======+======+======+=======+" << std::endl;
    while (!feof(info))
    {
        if (fscanf(info, "%s %d", filename, &refcount) != 2 || refcount < 0)
            break;

        image = cv::imread(fullname, cv::IMREAD_COLOR);
        if (image.empty())
        {
            std::cerr << "WARNING: Obrazok sa nepodarilo nacitat: " << fullname << std::endl;
            continue;
        }

        // nacitanie suradnic objektov vo vstupnom subore
        int x, y, w, h;
        ObjectPos tmp;
        std::vector<ObjectPos> ref;
        for (int i = 0; i < refcount; i++)
        {
            if (fscanf(info, "%d %d %d %d", &x, &y, &w, &h) != 4)
            {
                std::cerr << "ERROR: Nespravny format vstupneho suboru" << std::endl;
                return;
            }

            // vypocet stredu obdlznika
            tmp.x = 0.5F * w + x;
            tmp.y = 0.5F * h + y;
            // vypocet priemernej dlzky strany
            tmp.width = 0.5F * (w + h);
            tmp.found = false;
            ref.push_back(tmp);

            if (saveDetected)
                rectangle(image, cv::Rect(x, y, w, h), CV_RGB(0, 255, 0));
        }

        // spustenie detekcie na nacitanom obrazku
        cvtColor(image, grayImage, CV_BGR2GRAY);
        // ekvalizáciu histogramu ak je zadane -norm
        if (equalize)
        {
            cv::Mat temp;
            equalizeHist(grayImage, temp);
            grayImage = temp;
        }

        std::vector<cv::Rect> objects;
        totaltime -= (double)cvGetTickCount();
        cascade.detectMultiScale(grayImage, objects, scaleFactor, minNeighbors, 0, minSize, maxSize);
        totaltime += (double)cvGetTickCount();
        hits = missed = falseAlarms = 0;

        ObjectPos det;
        // meranie vykonosti
        for (int i = 0; i < objects.size(); i++)
        {
            ////PERFORMANCE SCALE
            //tmpw = objects[i].width / (float) 1.1; // sirka zmensena o 10%
            //tmph = objects[i].height / (float) 1.1; // vyska zmensena o 10%

            // vypocet stredu obdlznika
            det.x = 0.5F * objects[i].width + objects[i].x;
            det.y = 0.5F * objects[i].height + objects[i].y;
            // vypocet priemernej dlzky strany
            det.width = 0.5F * (objects[i].width + objects[i].height);

            ////PERFORMANCE SCALE
            //det.width = sqrtf( 0.5F * (tmpw * tmpw + tmph * tmph));		
            //// uplatnenie zmensenia aj do vykresleneho bb
            //objects[i].width = tmpw; // sirka zmensena o 10%
            //objects[i].height = tmph; // vyska zmensena o 10%
            //objects[i].x = (objects[i].x + (objects[i].width - tmpw) / (float)2 );
            //objects[i].y = (objects[i].y + (objects[i].height - tmph) / (float)2 );

            found = 0;
            for (int j = 0; j < refcount; j++)
            {
                float distance = sqrtf((det.x - ref[j].x) * (det.x - ref[j].x) +
                    (det.y - ref[j].y) * (det.y - ref[j].y));

                if ((distance < ref[j].width * maxPosDiff) &&
                    (det.width > ref[j].width - ref[j].width * maxSizeDiff) &&
                    (det.width < ref[j].width + ref[j].width * maxSizeDiff))
                {
                    ref[j].found = 1;
                    found = 1;
                    if (saveDetected)
                        rectangle(image, objects[i], CV_RGB(0, 0, 255), 2);
                }
            }

            if (!found)
                falseAlarms++;

            // ulozenie vysledku detekcie ak je zadane -save
            if (saveDetected && !found)
                rectangle(image, objects[i], CV_RGB(255, 0, 0));
        }

        for (int j = 0; j < refcount; j++)
        {
            if (ref[j].found)
                hits++;
            else
                missed++;
        }

        totalHits += hits;
        totalMissed += missed;
        totalFalseAlarms += falseAlarms;
        totalObjects += objects.size();
        printf("|%32.32s|%6d|%6d|%6d|%7lld|\n", filename, hits, missed, falseAlarms, objects.size());
        std::cout << "+--------------------------------+------+------+------+-------+" << std::endl;
        fflush(stdout);

        if (saveDetected)
        {
            strcpy(detfilename, detname);
            strcat(detfilename, filename);
            strcpy(filename, detfilename);
            imwrite(fullname, image);
        }
    }
    fclose(info);

    printf("|%32.32s|%6d|%6d|%6d|%7d|\n", "Total", totalHits, totalMissed, totalFalseAlarms, totalObjects);
    std::cout << "+--------------------------------+------+------+------+-------+" << std::endl;
    //printf( "Number of stages: %d\n",  );
    //printf( "Number of weak classifiers: %d\n", );
    printf("Celkovy cas detekcie: %g ms\n", totaltime / ((double)cvGetTickFrequency()*1000.));
}

void detectImage(int argc, char* argv[])
{
    std::string model = "cascade.xml";
    std::string inputImage = "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png";
    bool gpu = false;
    bool save = false;
    if (argc == 2)
    {
        std::cout << "Vizualizacia detekcie" << std::endl;
        std::cout << "Pouzitie: " << std::endl;
        std::cout << "  -classifier <classifier_file_path> - vstupny natrenovany klasifikator" << std::endl;
        std::cout << "  -input <image_file_path> - vstupny obrazok" << std::endl;
        std::cout << "  [-gpu] - model pre GPU aj CPU" << std::endl;
        std::cout << "  [-save] - ulozit na disk, inak zobrazit na obrazovku" << std::endl;
    }
    for (int i = 2; i < argc; i++)
    {
        if (!strcmp(argv[i], "-classifier"))
        {
            model = argv[++i];
        }
        else if (!strcmp(argv[i], "-input"))
        {
            inputImage = argv[++i];
        }
        else if (!strcmp(argv[i], "-gpu"))
        {
            gpu = true;
        }
        else if (!strcmp(argv[i], "-save"))
        {
            save = true;
        }
    }
    detection(model, inputImage, gpu, save);
}

/*
ocasovat cpu boost a gpu boost, trening a detekcia niekolkych obrazkov
*/

void commands(int argc, char* argv[], int defaultCommand = -1)
{
    if (defaultCommand == -1 && argc == 1)
    {
        printf("Vyberte moznost:\n");
        printf("0: Koniec.\n");
        printf("1: Test obrazku na natrenovanom modeli pre detekciu tvari.\n");
        printf("2: Test obrazku na natrenovanom modeli pre detekciu tiel.\n");
        printf("3: Trening modelu na pribalenych datach.\n");
        printf("4: Benchmark natrenovaneho modelu.\n");
        printf("5: Test natrenovaneho modelu na jednom obrazku.\n");
        printf("6: Prototyp vypoctu novych crt.\n");
        printf("7: Vytvorenie bg.txt pre negativne VJ sample.\n");
        printf("8: Vytvorenie info.dat pre negativne VJ sample.\n");
        printf("9: Test obrazku na natrenovanom modeli pre detekciu tvari. Len CPU.\n");
        printf("10: Test obrazku na natrenovanom modeli pre detekciu tiel. Len CPU.\n");
        printf("11: Test obrazku na vlastnom natrenovanom modeli pre detekciu hracov. Len CPU.\n");
        printf("12: Test obrazku na vlastnom neuplne natrenovanom modeli pre detekciu hracov. Len CPU.\n");
        printf("13: Statistika natrenovaneho modelu voci vstupnym datam. Len cez prikazovy riadok.\n");
        printf("14: Test obrazku na SHOG modeli pre detekciu hracov. Len CPU.\n");
        printf("15-19: Test obrazku na roznych modeloch. Len CPU.\n");
        printf("20: Test obrazku cez prikazovy riadok na zadanom obrazku a modeli.\n");
    }
    else if (argc > 1)
        defaultCommand = std::atoi(argv[1]);
    switch (defaultCommand)
    {
        case 1:
            detection("..\\XMLCuda\\haarcascade_frontalface_alt.xml", "..\\inputImages\\happypeople.jpg", true);
            break;
        case 2:
            detection("..\\XMLCuda\\haarcascade_fullbody.xml", "..\\inputImages\\SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png", true);
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
        case 6:
        {
            FeaturePrototypes::test();
            break;
        }
        case 7:
        {
            Parser parser;
            std::string directory;
            printf("Zadajte meno priecinku: ");
            std::getline(std::cin, directory);
            std::getline(std::cin, directory);
            std::string file("bg.txt");
            parser.MakeDatFile(directory, file);
            break;
        }
        case 8:
        {
            Parser parser;
            std::string directory;
            printf("Zadajte meno priecinku: ");
            std::getline(std::cin, directory);
            std::getline(std::cin, directory);
            std::vector<std::string> filenames;
            parser.GetFileNames(directory, filenames);
            std::ofstream outputFile;
            outputFile.open(("info.dat"));
            for (std::string& filename : filenames)
            {
                cv::Mat image = cv::imread((directory + "\\" + filename).data(), CV_LOAD_IMAGE_GRAYSCALE);
                outputFile << directory << "/" << filename << " 1 0 0 " << std::to_string(image.cols) << " " << std::to_string(image.rows) << std::endl;
            }
            break;
        }
        case 9:
            detection("..\\XMLCPU\\haarcascade_frontalface_default.xml", "..\\inputImages\\happypeople.jpg", false);
            break;
        case 10:
            detection("..\\XMLCPU\\haarcascade_fullbody.xml", "..\\inputImages\\SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png", false);
            break;
        case 11:
            detection("..\\XMLCPU\\cascade.xml", "..\\inputImages\\SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png", false);
            break;
        case 12:
            detection("..\\XMLCPU\\tempSave.xml", "..\\inputImages\\SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png", false);
            break;
        case 13:
            cascadePerformance(argc, argv);
            break;
        case 14:
            detection("..\\XMLCPU\\cascadeSHOG.xml", "..\\inputImages\\20170422135715126.Png", false);
            break;
        case 15:
            detection("..\\XMLCPU\\cascade2.xml", "..\\inputImages\\20170422135715126.Png", false);
            break;
        case 16:
            detection("..\\XMLCPU\\cascade2.xml", "..\\inputImages\\SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png", false);
            break;
        case 17:
            detection("..\\XMLCPU\\cascade2.xml", "..\\inputImages\\20160428114934750.Png", false);
            break;
        case 18:
            detection("..\\XMLCPU\\cascadeSHOG.xml", "..\\inputImages\\20160428114934750.Png", false);
            break;
        case 19:
            detection("..\\XMLCPU\\cascadeSHOG.xml", "..\\inputImages\\SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png", false);
            break;
        default:
            break;
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
    //std::vector<std::string> filenames;
    //Evaluator eval(filenames);
    //eval.trainint(false,"HaarXMLPrezentacia.xml",sampleFolders);
    //eval.detect(false, "HaarXMLPrezentacia.xml",sampleFolders);
    //eval.detectMultiScaleProto(false, "HaarXML2640.xml", "outputHaar1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png");
    //clock_t end = clock();
    //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //printf("Seconds %lf\n", elapsed_secs);
    commands(argc, argv, 6);

    //std::chrono::time_point<std::chrono::system_clock> start, end;
    //start = std::chrono::system_clock::now();
    //eval.detectMultiScaleProto(false, "HaarXML2640.xml", "outputHaar1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png");
    //detection("haarcascade_fullbody.xml", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002581.png");
    //end = std::chrono::system_clock::now();

    //std::chrono::duration<double> elapsed_seconds = end - start;
    //std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    //std::cout << "finished computation at " << std::ctime(&end_time)
    //    << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return 0;
}