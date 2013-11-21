#include <stdio.h>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\contrib\contrib.hpp"

#include <Windows.h>

using namespace cv;
using namespace std;

double NccCoreCalculation(Mat &regionLeft, Mat &regionRight, double d)
{
	unsigned short numerator = 0;
	double denominator = 0.0;
	unsigned short denominatorRight = 0;
	unsigned short denominatorLeft = 0;

	uchar *r, *l;
	for (int i = 0; i < regionLeft.rows; i++)
	{
		l = regionLeft.ptr<uchar>(i);
		r = regionRight.ptr<uchar>(i);
		for (int j = 0; j < regionLeft.cols; j++)
		{
			numerator += r[j] * l[j];
			denominatorLeft += (l[j] ^ 2);
			denominatorRight += (r[j] ^ 2);
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

int GetDisparity(vector<int> disparitiesToSearch,int jWinStr,
	int iWinStr, int winx,	int winy, Mat &image,Mat &regionTemplate)
{
	//Returns the disparity of a region

	Rect regionToMatch;
	Mat regionToMatchMat;
	double ncc = 0.0;
	double previousCorrelation = 0.0;
	int bestMatchSoFar = 0;
	for (int i = 0; i < disparitiesToSearch.size(); i++)
	{
		regionToMatch = Rect(jWinStr + disparitiesToSearch[i], iWinStr, winx, winy);
		regionToMatchMat = image(regionToMatch);
		ncc = NccCoreCalculation(regionToMatchMat, regionTemplate, disparitiesToSearch[i]);
		if (ncc > previousCorrelation)
		{
			previousCorrelation = ncc;
			bestMatchSoFar = disparitiesToSearch[i];
		}
	}
	return bestMatchSoFar;
}


Mat NCCSR(Mat imL, Mat imR, int windowSize, int dispMin, int dispMax)
{
	// pad the left and right images
	copyMakeBorder(imL, imL, 0, 0, dispMax, dispMax,BORDER_CONSTANT, Scalar(0));
	copyMakeBorder(imR, imR, 0, 0, dispMax, dispMax, BORDER_CONSTANT, Scalar(0));

	int nrLeft = imL.rows;
	int ncLeft = imL.cols;

	int winx = windowSize - 1;
	int winy = (windowSize - 1) / 2;

	int bottomLine = nrLeft - winy;
	int searchRange = 3;
	int searchWidth = round(searchRange * 0.5);

	Mat NREDCRef = Mat::ones(3, ncLeft, imL.type());

	Mat dispMap = Mat::zeros(nrLeft, ncLeft, imL.type());

	uchar *dispMapRowPointer;

	int i = nrLeft - winy;
	for (i ; i > winy; i--)
	{
		dispMapRowPointer = dispMap.ptr<uchar>(i);

		int iWinStr = i - winy;
		int iWinEnd = i + winy;

		for (int j = 1 + winx + dispMax; j < ncLeft - winx - dispMax; j++)
		{
			int jWinStr = j - winx;
			int jWinEnd = j + winx;

			Rect regionRight = Rect(jWinStr, iWinStr, winx, winy);
			Mat regionRightMat = imR(regionRight);

			vector<int> disparitiesToSearch;

			if (i == bottomLine)
			{
				for (int d = dispMin; d <= dispMax; d++)
					disparitiesToSearch.push_back(d);
			}
			else 
			{
				/*for (int d = dispMin; d <= dispMax; d++)
					disparitiesToSearch.push_back(d);*/
				

				for (int k = 0; k < NREDCRef.rows; k++)
				{
					uchar disparitiesToSearch[9];
					
					uchar val = NREDCRef.at<uchar>(Point(j, k));
					uchar val1 = NREDCRef.at<uchar>(Point(j - 1,k));
					uchar val2 = NREDCRef.at<uchar>(Point( j + 1,k));

					disparitiesToSearch.push_back(val);
					disparitiesToSearch.push_back(val1);
					disparitiesToSearch.push_back(val2);
				}
				//disparitiesToSearch = GetUniqueAndValidDisparities(disparitiesToSearch,dispMax);
			}

			dispMapRowPointer[j] = GetDisparity(disparitiesToSearch,
				jWinStr, iWinStr, winx, winy, imL, regionRightMat);
		}

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

	for (int i = 0 + win; i< heightY - win; i++)
	{
		d = dispMap.ptr<uchar>(i);

		for (int j = 0 + win + dispMax; j< widthX - win; j++)
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

		double startTime = getTickCount();

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
	


	for (int i = 0; i < numIterations; i++){
		double startTime = getTickCount();

		Mat dispMap = NCCSR(left, right, 11, 0, 100);
		
		double iterationTime = (getTickCount() - startTime) / getTickFrequency();

		//cout << "Iteration #  " << i + 1 << " in " << iterationTime << " s" << endl;

		totalTime += iterationTime;
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

int main()
{
	int iterations = 2;

	Mat left = imread("left1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat right = imread("right1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	

	double time = GetAverageImageProcessingTime(left, right, iterations);
	cout << "Average time of:  " << time << "s over " << iterations << " iterations" << endl;
	
	left = imread("left2.png", CV_LOAD_IMAGE_GRAYSCALE);
	right = imread("right2.png", CV_LOAD_IMAGE_GRAYSCALE);


	time = GetAverageImageProcessingTime(left, right, iterations);
	cout << "Average time of:  " << time << "s over " << iterations << " iterations" << endl;


	left = imread("left3.png", CV_LOAD_IMAGE_GRAYSCALE);
	right = imread("right3.png", CV_LOAD_IMAGE_GRAYSCALE);


	time = GetAverageImageProcessingTime(left, right, iterations);
	cout << "Average time of:  " << time << "s over " << iterations << " iterations" << endl;

	WaitKey();
	return 0;
}

