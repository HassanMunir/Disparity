#include <stdio.h>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "Intrin.h"
#include <omp.h>
#include "Timer.h"

#include <Windows.h>

using namespace cv;
using namespace std;

int _segmentSize = 25;

double NccCoreCalculation(Mat &regionLeft, Mat &regionRight, double d)
{
    double numerator = 0;
    double denominator = 0.0;
    double denominatorRight = 0;
    double denominatorLeft = 0;

    uchar *r, *l;
    for (int i = 0; i < regionLeft.rows; i++)
    {
        l = regionLeft.ptr<uchar>(i);
        r = regionRight.ptr<uchar>(i);
        for (int j = 0; j < regionLeft.cols; j++)
        {

            //numerator += __emul(r[j], l[j]);
            numerator += r[j] * l[j];
            denominatorLeft += l[j] * l[j];
            denominatorRight += r[j] * r[j];
        }
    }
    denominator = sqrt(denominatorLeft * denominatorRight);
    double ncc = numerator / denominator;
    return ncc;

}

vector<int> GetUniqueAndValidDisparities(vector<int> disparitiesToSearch, int dispMax)
{
    vector<int> results;
    sort(disparitiesToSearch.begin(), disparitiesToSearch.end());

    results.push_back(disparitiesToSearch[0]);
    int resultsCount = 0;
    for (int i = 1; i < disparitiesToSearch.size(); i++)
    {
        if ((disparitiesToSearch[i] != results[resultsCount]) &&
            (disparitiesToSearch[i] >= 0) &&
            (disparitiesToSearch[i] <= dispMax))
        {
            results.push_back(disparitiesToSearch[i]);
            resultsCount++;
        }
    }
    return results;

}

int _countIterations = 0;
double _totalTime = 0.0;

int GetDisparity(int disparitiesToSearch [], int numDisp, int jWinStr,
    int iWinStr, int winx, int winy, Mat &image, Mat &regionTemplate)
{
    //Returns the disparity of a region
    //_countIterations++;

    double previousCorrelation = 0.0;
    int bestMatchSoFar = 0;

    vector<double> ncc_vector;
    ncc_vector.resize(numDisp);

    //Timer timer = Timer("NCC");

#pragma omp parallel num_threads(1) shared(numDisp, jWinStr, disparitiesToSearch, iWinStr, winx, winy,ncc_vector)
    {
#pragma omp for
        for (int k = 0; k < numDisp; k++)
        {
            //cout << i << endl;

            Rect regionToMatch = Rect(jWinStr + disparitiesToSearch[k], iWinStr, winx, winy);
            Mat regionToMatchMat = image(regionToMatch);

            //double ncc = NccCoreCalculation(regionToMatchMat, regionTemplate, disparitiesToSearch[i]);

            double numerator = 0;
            double denominator = 0.0;
            double denominatorRight = 0;
            double denominatorLeft = 0;

            uchar *r, *l;
            for (int i = 0; i < regionToMatchMat.rows; i++)
            {
                l = regionToMatchMat.ptr<uchar>(i);
                r = regionTemplate.ptr<uchar>(i);
                for (int j = 0; j < regionToMatchMat.cols; j++)
                {
                    //numerator += __emul(r[j], l[j]);
                    numerator += r[j] * l[j];
                    denominatorLeft += l[j] * l[j];
                    denominatorRight += r[j] * r[j];
                }
            }
            denominator = sqrt(denominatorLeft * denominatorRight);

            ncc_vector[k] = numerator / denominator;

        }
    }

    //_totalTime += timer.stop();

    //Timer timer1 = Timer("Finding best match");
    double prev = ncc_vector[0];
    bestMatchSoFar = disparitiesToSearch[0];

    for (int k = 1; k < ncc_vector.size(); k++)
    {
        if (ncc_vector[k] > prev)
        {
            prev = ncc_vector[k];
            bestMatchSoFar = disparitiesToSearch[k];
        }
    }
    //timer.stop();

    return bestMatchSoFar;
}

Mat NCCSR(Mat imL, Mat imR, int windowSize, int dispMin, int dispMax, int disparitiesToSearch [])
{
    // pad the left and right images
    copyMakeBorder(imL, imL, 0, 0, dispMax, dispMax, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(imR, imR, 0, 0, dispMax, dispMax, BORDER_CONSTANT, Scalar(0));

    int nrLeft = imL.rows;
    int ncLeft = imL.cols;

    int winx = windowSize - 1;
    int winy = (windowSize - 1) / 2;

    int bottomLine = nrLeft - winy;
    int searchRange = 3;
    int searchWidth = (int) round(searchRange * 0.5);

    Mat NREDCRef = Mat::ones(3, ncLeft, imL.type());

    Mat dispMap = Mat::zeros(nrLeft, ncLeft, imL.type());

    uchar *dispMapRowPointer;

    int i = nrLeft - winy;
    for (i; i > winy; i--)
    {
        dispMapRowPointer = dispMap.ptr<uchar>(i);

        int iWinStr = i - winy;
        int iWinEnd = i + winy;
        int startingPixel= 1 + winx + dispMax;
        int endPixel = ncLeft - winx - dispMax;

#pragma omp parallel num_threads(2) shared(dispMax, winx, winy, disparitiesToSearch, imR, imL)
        {
#pragma omp for
            //Timer timer = Timer("Line");
            for (int j = startingPixel; j < endPixel; j++)
            {
                int jWinStr = j - winx;
                int jWinEnd = j + winx;

                Rect regionRight = Rect(jWinStr, iWinStr, winx, winy);
                Mat regionRightMat = imR(regionRight);
                
                if (i == bottomLine && (i % _segmentSize == 0))
                {
                    dispMapRowPointer[j] = GetDisparity(disparitiesToSearch, dispMax,
                        jWinStr, iWinStr, winx, winy, imL, regionRightMat);
                }
                else
                {
                    /*for (int d = dispMin; d <= dispMax; d++)
                        disparitiesToSearch.push_back(d);*/

                    int disparitiesToSearch[9];

                    uchar *NREDCRefPointer = NREDCRef.ptr<uchar>(0);
                    disparitiesToSearch[0] = NREDCRefPointer[j];
                    disparitiesToSearch[1] = NREDCRefPointer[j - 1];
                    disparitiesToSearch[2] = NREDCRefPointer[j + 1];

                    NREDCRefPointer = NREDCRef.ptr<uchar>(1);
                    disparitiesToSearch[3] = NREDCRefPointer[j];
                    disparitiesToSearch[4] = NREDCRefPointer[j - 1];
                    disparitiesToSearch[5] = NREDCRefPointer[j + 1];

                    NREDCRefPointer = NREDCRef.ptr<uchar>(2);
                    disparitiesToSearch[6] = NREDCRefPointer[j];
                    disparitiesToSearch[7] = NREDCRefPointer[j - 1];
                    disparitiesToSearch[8] = NREDCRefPointer[j + 1];

                    dispMapRowPointer[j] = GetDisparity(disparitiesToSearch, 9,
                        jWinStr, iWinStr, winx, winy, imL, regionRightMat);
                    //disparitiesToSearch = GetUniqueAndValidDisparities(disparitiesToSearch,dispMax);
                }
            }

        }
        //timer.stop();

        dispMap.row(i).copyTo(NREDCRef.row(1));
        NREDCRef.row(0) = NREDCRef.row(1) - 1;
        NREDCRef.row(2) = NREDCRef.row(1) + 1;
    }

    //Remove the padding
    Rect dispMapRegion = Rect(dispMax, 0, imL.cols - 3 * dispMax, imL.rows);
    dispMap = dispMap(dispMapRegion);

    return dispMap;
}

void PrepareMapForDisplay(Mat &dispMap)
{
    double minVal, maxVal;
    cv::minMaxLoc(dispMap, &minVal, &maxVal, 0, 0);
    double factor = 255.0 / (maxVal - minVal);
    dispMap = dispMap * factor;
    applyColorMap(dispMap, dispMap, COLORMAP_JET);
}

Mat NCC(Mat imL, Mat imR, int windowSize, int dispMax)
{
    int dispMin = 0;

    int heightY = imL.rows;
    int widthX = imL.cols;

    if (windowSize % 2 == 0){
        cerr << "'The window size must be an odd number.'";
    }

    if (dispMin > dispMax)
    {
        cerr << "'Minimum Disparity must be less than the Maximum disparity.'";
    }


    Mat dispMap = Mat::zeros(heightY, widthX, imR.type());

    uchar *l, *r, *d;

    int win = (windowSize - 1) / 2;

    for (int i = 0 + win; i < heightY - win; i++)
    {
        d = dispMap.ptr<uchar>(i);

        for (int j = 0 + win + dispMax; j < widthX - win; j++)
        {
            double prevNCC = 0.0;
            int bestMatchSoFar = dispMin;
            for (int dispRange = dispMin; dispRange < dispMax; dispRange++)
            {
                double nccNumerator = 0.0;
                double nccDenominator = 0.0;
                double nccDenominatorRightWindow = 0.0;
                double nccDenominatorLeftWindow = 0.0;
                for (int a = -win; a < win; a++)
                {
                    l = imL.ptr<uchar>(i + a);
                    r = imR.ptr<uchar>(i + a);
                    for (int b = -win; b < win; b++)
                    {
                        //nccNumerator = nccNumerator+(rightImage
                        nccNumerator = nccNumerator + (r[j + b - dispRange] * l[j + b]);
                        nccDenominatorRightWindow += (r[j + b - dispRange] * r[j + b - dispRange]);
                        nccDenominatorLeftWindow += (l[j + b] * l[j + b]);
                    }
                }
                nccDenominator = sqrt(nccDenominatorRightWindow*nccDenominatorLeftWindow);
                double ncc = nccNumerator / nccDenominator;
                if (prevNCC < ncc)
                {
                    prevNCC = ncc;
                    bestMatchSoFar = dispRange;
                }
            }
            //dispMap = bestMatchSoFar;
            d[j] = bestMatchSoFar;
        }
    }
    return dispMap;
}

void ProcessFolder(string inputFolder, string outputFolder)
{
    std::cout << "Hello" << std::endl;

    WIN32_FIND_DATA search_data;
    memset(&search_data, 0, sizeof(WIN32_FIND_DATA));

    HANDLE handle = FindFirstFile("SmallSample\\*.png", &search_data);

    bool loadLeft = true;
    bool loadRight = false;

    vector<string> leftFiles;
    vector<string> rightFiles;

    while (handle != INVALID_HANDLE_VALUE)
    {
        if (search_data.cFileName[1] == '1')
            leftFiles.push_back(search_data.cFileName);
        else
            rightFiles.push_back(search_data.cFileName);

        if (FindNextFile(handle, &search_data) == FALSE)
            break;
    }

    for (int i = 0; i < leftFiles.size(); i++)
    {
        cout << "left filename: \t"
            << leftFiles[i]
            << endl;


        cout << "right filename: \t"
            << rightFiles[i]
            << endl;

        Mat left = imread("SmallSample\\" + leftFiles[i], CV_LOAD_IMAGE_GRAYSCALE);
        Mat right = imread("SmallSample\\" + rightFiles[i], CV_LOAD_IMAGE_GRAYSCALE);

        if (!left.data || !right.data)
        {
            std::cerr << "Count not open or find the image" << std::endl;
            return;
        }

        int64 startTime = getTickCount();

        //Mat dispMap = NCCSR(left, right, 11, 0, 100);
        Mat dispMap = NCC(left, right, 11, 100);

        double timeTaken = (getTickCount() - startTime) / getTickFrequency();

        PrepareMapForDisplay(dispMap);

        string extension = ".png";
        string filename = static_cast<ostringstream*>(&(ostringstream() << i))->str();
        string outputFile = "Output\\NCC\\" + filename + extension;
        cout << "saved as:" << outputFile << endl;
        imwrite(outputFile, dispMap);
    }

}

double GetAverageImageProcessingTime(Mat left, Mat right, int numIterations)
{
    if (!left.data || !right.data)
    {
        std::cerr << "Count not open or find the image" << std::endl;
        return -1;
    }

    double totalTime = 0.0;

    int dispMax = 100;

    int disparitiesToSearch[100];

    for (int d = 0; d < dispMax; d++)
        disparitiesToSearch[d] = d;


    for (int i = 0; i < numIterations; i++){
        int64 startTime = getTickCount();

        Mat dispMap = NCCSR(left, right, 11, 0, dispMax, disparitiesToSearch);

        double iterationTime = (getTickCount() - startTime) / getTickFrequency();

        //cout << "Iteration #  " << i + 1 << " in " << iterationTime << " s" << endl;

        totalTime += iterationTime;

        //PrepareMapForDisplay(dispMap);
        //imshow("dispMap", dispMap);
        //waitKey(0);
    }

    double timeTaken = totalTime / numIterations;

    return timeTaken;
}

void WaitKey()
{
    cout << "Enter any character followed by the 'Enter' key to exit..";
    string input;
    cin >> input;
}

vector<Mat> SegmentImage(Mat image, int segmentHeight, int winSize)
{
    vector<Mat> segments;
    Rect segment;

    int rows = image.rows;
    int numSegments = rows / segmentHeight;

    for (int i = 0; i < numSegments; i++)
    {
        if (i == 0)
        {
            segment = Rect(0, i*segmentHeight, image.cols, segmentHeight + winSize);
        }
        else if (i == numSegments - 1)
        {
            segment = Rect(0, i* segmentHeight - winSize, image.cols, segmentHeight);
        }
        else {
            segment = Rect(0, i*segmentHeight - (2 * winSize), image.cols, segmentHeight + (winSize * 2));
        }

        segments.push_back(image(segment));
    }
    return segments;
}


int main()
{
    int iterations = 1;

    Mat left = imread("left1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat right = imread("right1.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    vector<Mat> leftSegments = SegmentImage(left, _segmentSize, 11);
    vector<Mat> rightSegments = SegmentImage(right, _segmentSize, 11);

    int disparitiesToSearch[100];
    int dispMax = 100;

    for (int d = 0; d < dispMax; d++)
        disparitiesToSearch[d] = d;

    Mat disp;

    Timer timer = Timer("Disparity Calculation");
    disp = NCCSR(left, right, 11, 0, dispMax, disparitiesToSearch);
    timer.print();

    //cout << "Average per line: " << (double) _totalTime / _countIterations << endl;


    imshow("disp", disp);
    waitKey(0);

    //	vector<Mat> disparities;
    //
    //	int64 startTime = getTickCount();
    //

    //
    //
    //#pragma omp parallel for
    //	for (int i = 0; i < leftSegments.size(); i++)
    //	{
    //		Mat disp = NCCSR(leftSegments[i], rightSegments[i], 11, 0, dispMax, disparitiesToSearch);
    //		//cout << "hello from: " << omp_get_thread_num() << endl;
    //		//disparities.push_back(disp);
    //	}
    //
    //	double finalTime = (getTickCount() - startTime) / getTickFrequency();
    //
    //	cout << finalTime << endl;
    //
    //
    //	//for (int i = 0; i < disparities.size(); i++)
    //	//{
    //	//	PrepareMapForDisplay(disparities[i]);
    //	//	imshow("disp" + i, disparities[i]);
    //	//}
    //
    //	//waitKey(0);
    //
    //	double time = GetAverageImageProcessingTime(left, right, iterations);
    //	cout << "Average time of:  " << time << "s over " << iterations << " iterations" << endl;
    //
    //	//left = imread("left2.png", CV_LOAD_IMAGE_GRAYSCALE);
    //	//right = imread("right2.png", CV_LOAD_IMAGE_GRAYSCALE);
    //
    //
    //	//time = GetAverageImageProcessingTime(left, right, iterations);
    //	//cout << "Average time of:  " << time << "s over " << iterations << " iterations" << endl;
    //
    //
    //	//left = imread("left3.png", CV_LOAD_IMAGE_GRAYSCALE);
    //	//right = imread("right3.png", CV_LOAD_IMAGE_GRAYSCALE);
    //
    //
    //	//time = GetAverageImageProcessingTime(left, right, iterations);
    //	//cout << "Average time of:  " << time << "s over " << iterations << " iterations" << endl;
    //
    //WaitKey();
    return 0;
}

