#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>     //std::fixed
#include <iomanip>      // std::setprecision
#include <RunningStats.h>
#include <chrono>
#include <string>

//face detection
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>

//countour detection
#include <stdlib.h>

//file output
#include <fstream>      // std::fstream

using namespace std;
using namespace cv;
using namespace saliency;
using std::fstream;

/** Function Headers */
void detectFacesAndDisplay(const Mat& frame, const Mat& image);
static void detectContoursAndDisplay(const Mat& img, const Mat& dst);
static void processColors(const Mat& colorMat);
CascadeClassifier face_cascade, body_cascade;
// running statistics class is used to compute statistics in one pass trough the data
RunningStats runstatStaticSaliency, runstatMotionSaliency, runstatContourArea,
runstatContours, runstatMultipleFacesMean, runstatColourfull;
double totalMotionSaliency, totalFaceBool, totalBodyBool, totalFaceMultiple, oldMotionValue;
double thresh = 1;
RNG rng(12345);
bool method = false;
bool preview = false;
//bool objectness = false;
bool drawFaces = false;
bool drawTheContours = false;
int resizeMode = 1;
Size maxFaceSize = Size(30, 30);
double facePercent = 0.0;
Size frameResizeSize = Size(320, 240);
int cuts = 0;
double finalAttentionScore = 0.0;
double frameContourArea = 0.0;
int bw = 0;
double colourFull = 0.0;

static const char* keys =
{ "{@video_name      | | video name            }"
"{@preview    |0| preview mode}" };

static void help()
{
	cout << "\nThis utility processes videos and return an attention score [0,1]"
		"Call:\n"
		"./attentionScore.exe  <video_name> <preview mode> \n"
		<< endl;
}

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);
	String saliency_algorithm = "BinWangApr2014";
	String video_name = parser.get<String>(0);
	//resizeMode = parser.get<int>(1);
	//String training_path = parser.get<String>(2);

	if (parser.get<String>(1) == "1") {
		//system("pause");
		preview = true;
	}
	String saliency_algorithm2 = "SPECTRAL_RESIDUAL";
	String saliency_algorithm3 = "FINE_GRAINED";
	//String saliency_algorithm4 = "BING"; //for objectness
	String face_cascade_name = "haar/haarcascade_frontalface_alt.xml";
	//String body_cascade_name = "haar/haarcascade_mcs_upperbody.xml";

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	//if (!body_cascade.load(body_cascade_name)) { printf("--(!)Error loading body cascade\n"); return -1; };

	//Capture the video
	VideoCapture cap;

	if (video_name.length() == 1) {        //if the first argument have size 1
		cap.open(stoi(video_name));      // try to open camera device number (e.g:0,1..)
	}
	else {                                //else try to open a video file
		cap.open(video_name);
		cap.set(CAP_PROP_POS_FRAMES, 1);

	}

	if (!cap.isOpened())
	{
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}

	Mat image;
	Mat frame;
	//Mat image2;
	Mat motionSaliencyMap;

	//instantiates the specific Saliency
	Ptr<Saliency> motionSaliencyAlgorithm = Saliency::create(saliency_algorithm);

	String choosenMethod = "";
	if (!method) { choosenMethod = saliency_algorithm2; }
	else { choosenMethod = saliency_algorithm3; }

	Ptr<Saliency> staticSaliencyAlgorithm = Saliency::create(choosenMethod);

	//Ptr<Saliency> saliencyAlgorithm4 = Saliency::create(saliency_algorithm4); //for objectness

	if (motionSaliencyAlgorithm == NULL)
	{
		cout << "***Error in the instantiation of the saliency algorithm...***\n";
		return -1;
	}

	cap >> frame;
	if (frame.empty())
	{
		return 0;
	}
	frame.copyTo(image);
	//frame.copyTo(image2);
	int frameCount = 0;

	motionSaliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->setImagesize(image.cols, image.rows);
	motionSaliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->init();

	//for objectness
	//vector<Vec4i> objVecMap;
	//saliencyAlgorithm4.dynamicCast<ObjectnessBING>()->setTrainingPath(training_path);
	//saliencyAlgorithm4.dynamicCast<ObjectnessBING>()->setBBResDir(training_path + "/Results");

	double totalPixels = (image.cols * image.rows);
	totalMotionSaliency = totalFaceBool = totalBodyBool = totalFaceMultiple = oldMotionValue = 0.0;
	runstatMotionSaliency.Clear();
	runstatMultipleFacesMean.Clear();
	runstatStaticSaliency.Clear();
	runstatContourArea.Clear();
	runstatContours.Clear();
	runstatColourfull.Clear();
	bool paused = false;

	switch (resizeMode)    //resize for performance sake
	{
	case 1:  //default
		frameResizeSize = Size(320, 240);
		break;
	case 2:
		frameResizeSize = Size(480, 360);
		break;
	case 3:
		frameResizeSize = Size(640, 480);
		break;
	}

	for (;; ) //forever loop
	{
		if (!paused)
		{
			Mat frameTemp;

			cap >> frameTemp; // capture each video frame

			if (frameTemp.empty()) {
				cout << endl;
				cout << " video file:      " << video_name << endl;
				cout << std::setprecision(4) << std::fixed <<
					" Static saliency: " << runstatStaticSaliency.Mean() << endl;

				cout << " Motion saliency: " << runstatMotionSaliency.Mean() << endl;
				cout << " Faces:           " << facePercent << endl;
				cout << " Cuts:            " << cuts - 2 << endl;
				cout << " Contour area:    " << runstatContourArea.Mean() * 30 / (totalPixels) << endl;
				cout << " Contour count:   " << runstatContours.Mean() << endl;
				cout << " Colourfulness:   " << colourFull << endl;
				cout << " Attention Score: " << finalAttentionScore << "\n"<<endl;
				//system("pause"); //will pause after program completion

				//Save metadata to the output csv file
				string outfile = "results/score_output.csv";
				//string destName = video_name.substr(video_name.find_last_of('/') + 1, video_name.size());
				//string finalName = "results/"+destName.substr(0, destName.find_last_of('.'))+".csv";
				ofstream myfile(outfile.c_str(), std::ofstream::out | std::ofstream::app);

				if (myfile.is_open()) {
					myfile << video_name << ",";
					myfile << runstatStaticSaliency.Mean() << ",";
					myfile << runstatMotionSaliency.Mean() << ",";
					myfile << facePercent << ",";
					myfile << cuts-2 << ",";
					myfile << runstatContourArea.Mean() * 30 / (totalPixels) << ",";
					myfile << runstatContours.Mean() << ",";
					myfile << colourFull << ",";
					myfile << finalAttentionScore;
					myfile << "\n";
					myfile.close();
				}

				exit(1);
			}

			if (resizeMode != 0) {
				resize(frameTemp, frame, frameResizeSize, 0, 0, INTER_NEAREST);
				image = frame.clone();
				cvtColor(frame, frame, COLOR_BGR2GRAY);
				//image2 = frame.clone();   //for objectness							
			}
			else {

				image = frameTemp.clone();
				cvtColor(frameTemp, frameTemp, COLOR_BGR2GRAY);
				frame = frameTemp;
			}
			frameCount++;

			//Motion saliency BingWangApr2014
			motionSaliencyAlgorithm->computeSaliency(frame, motionSaliencyMap);

			//Static saliency spectral residual or fine grained
			Mat staticSaliencyMap;
			staticSaliencyAlgorithm->computeSaliency(frame, staticSaliencyMap);

			/* for binary map
			Mat binaryMap;
			StaticSaliencySpectralResidual spec;
			spec.computeBinaryMap(staticSaliencyMap, binaryMap);
			*/
			//saliencyAlgorithm4->computeSaliency(image2, objVecMap); //for objectness

			//Process oponent colours
			processColors(image);
			//-- 3. Apply face detection to the frame
			detectFacesAndDisplay(frame, image);
			//-- 3. Apply contour detection to the saliency map
			detectContoursAndDisplay(staticSaliencyMap, image);

			//Let's compute percentage of moving pixels from the total frame pixels
			double whitePixelsCount = countNonZero(motionSaliencyMap);
			double frameBlackPixelsPercent = (totalPixels - whitePixelsCount) / totalPixels;
			double frameWhitePixelsPercent = 1 - frameBlackPixelsPercent;

			//Estimation of motion difference between subsequent frames
			double motionFrameDiff = abs(frameWhitePixelsPercent - oldMotionValue);
			oldMotionValue = frameWhitePixelsPercent;
			if (motionFrameDiff > 0.05) { cuts++; }

			//Calculate mean and std. deviance for static and motion saliency
			Scalar tempVal = mean(staticSaliencyMap);
			double myMAtMean = tempVal.val[0] * 3;
			if (isnan(myMAtMean))myMAtMean = 0;
			runstatStaticSaliency.Push(myMAtMean);

			double s2 = runstatStaticSaliency.StandardDeviation();
			double m2 = runstatMotionSaliency.StandardDeviation();
			if (isnan(s2))s2 = 0;
			if (isnan(m2))m2 = 0;

			facePercent = totalFaceBool / frameCount;

			if (bw < 100) {   //black and white detection
			colourFull = runstatColourfull.Mean();
			}
			else { colourFull = 0.0; }

			if (frameCount > 80) runstatMotionSaliency.Push(frameWhitePixelsPercent); //*3

			// Ad-hoc final metric depends on 4 metrics (static, motion saliency, faces, colourfulness)
			finalAttentionScore = (runstatMotionSaliency.Mean()+ (facePercent / 2) +
				((runstatContourArea.Mean() * 30) / totalPixels) + colourFull) / 4;
			String fas = "Score: " + std::to_string(finalAttentionScore);

			if (preview) {
				Point2f a(0, 0), b(140, 20);
				rectangle(image, a, b, (0, 0, 0), CV_FILLED, 8, 0);
				putText(image, fas, Point(5, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, 8, false);
				imshow("Original Image", image);
				//imshow("BING ojectness", image2);
				imshow("Motion Map", motionSaliencyMap);
				imshow("Static map", staticSaliencyMap);
			}

			cout << "\r" << "frames: " << frameCount << " Score: " << finalAttentionScore;

			motionSaliencyMap.release();
			staticSaliencyMap.release();
		}

		//keyboard shortcuts
		char c = (char)waitKey(2);
		if (c == '1')
			drawFaces = !drawFaces;

		if (c == '2')
			drawTheContours = !drawTheContours;

		if (c == 'q')
			break;

		if (c == 'p')
			paused = !paused;
	}

	return 0;
}

/** @function detectFacesAndDisplay */
void detectFacesAndDisplay(const Mat& frame, const Mat& image)
{
	std::vector<Rect> faces;
	equalizeHist(frame, frame);

	//-- Detect faces
	face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, maxFaceSize);

	if (faces.size())totalFaceBool++; //increment faces variable

	if (drawFaces) {
		for (size_t i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(image, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 255, 255), 2, 8, 0);
		}
	}
	runstatMultipleFacesMean.Push(faces.size()); //sum faces found to total

}
/** @function detectContoursAndDisplay */
static void detectContoursAndDisplay(const Mat& img, const Mat& dst) {

	//Convert one channel 8-bit matrix to 3 channel for canny input
	Mat fin_img;
	vector<Mat> channels;
	channels.push_back(img);
	channels.push_back(img);
	channels.push_back(img);
	merge(channels, fin_img);
	Mat B;
	fin_img.convertTo(B, CV_8U);

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	blur(B, B, Size(3, 3)); // first blur a little bit
	Canny(B, canny_output, thresh, thresh * 2, 3, true); //apply canny 
	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	double area0 = 0.0;

	for (int i = 0; i < contours.size(); i++)
	{
		vector<Point> contour = contours[i];
		area0 += contourArea(contour);

		if (drawTheContours) { 	/// Draw the contours
			Scalar color = Scalar(rng.uniform(0, 0), rng.uniform(150, 255), rng.uniform(0, 255));
			drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
		}
	}
	runstatContourArea.Push(area0);
	runstatContours.Push(contours.size());
}
/** @function processOponentColours */

static void processColors(const Mat& colorMat) {

	Scalar colAvg, colStds;
	meanStdDev(colorMat, colAvg, colStds);

	// C++ algorithm implementation of the
	// "Measuring colourfulness in natural images"
	// (Hasler and Susstrunk, 2003)

	double B = colAvg[0];
	double G = colAvg[1];
	double R = colAvg[2];

	//cout  << std::setprecision(4) << std::fixed << colAvg[0] << " " << colAvg[1] << " " << colAvg[2] <<endl;
	//cout << fabs(B - G) << " "<<fabs(G - R) << endl;

	double bg = fabs(B - G);
	double gr = fabs(G - R);

	if ((bg != 0.0) && (gr != 0.0)) { //if it is not black and white video

		double RGmean = colAvg[2] - colAvg[1];
		double YBmean = ((RGmean) / 2) - colAvg[0];
		double RGstd = colStds[2] - colStds[1];
		double YBstd = ((RGstd) / 2) - colStds[0];
		double colorfullnessMean = sqrt(pow(RGmean, 2) + pow(YBmean, 2));
		double colorfullnessStd = sqrt(pow(RGstd, 2) + pow(YBstd, 2));
		double ColorFull = colorfullnessStd + 0.3*colorfullnessMean;
		runstatColourfull.Push(ColorFull / 200);

	}
	else { bw++; }
}
