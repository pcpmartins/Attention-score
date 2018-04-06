#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>     //std::fixed
#include <iomanip>      // std::setprecision
#include <RunningStats.h>
#include <chrono>

//face detection
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>

//countour detection
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace saliency;

/** Function Headers */
void detectFacesAndDisplay(const Mat& frame, const Mat& image);
static void detectContoursAndDisplay(const Mat& img, const Mat& dst);
CascadeClassifier face_cascade, body_cascade;
// running statistics class is used to compute statistics in one pass trough the data
RunningStats runstatStaticSaliency, runstatMotionSaliency, runstatContourArea,
runstatContours, runstatMultipleFacesMean;
double totalMotionSaliency, totalFaceBool, totalBodyBool, totalFaceMultiple, oldMotionValue;
//StaticSaliencySpectralResidual spec;
double thresh = 1;
RNG rng(12345);
bool method = false;
bool objectness = false;
bool drawFaces = true;
bool drawTheContours = true;
int resizeMode = 0;
Size maxFaceSize = Size(10, 10);
Size maxBodySize = Size(10, 10);
Size frameResizeSize = Size(320, 240);
int cuts = 0;
double finalAttentionScore = 0.0;
double frameContourArea = 0.0;

static const char* keys =
{ "{@video_name      | | video name            }"
"{@resize_mode     |1| resize mode           }"
"{@training_path   |1| Path of the folder containing the trained files}" };

static void help()
{
	cout << "\nThis utility processes videos and return an attention score [0,1]"
		"Call:\n"
		"./attentionScore.exe  <video_name> <resize mode> \n"
		<< endl;
}

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);
	//String saliency_algorithm = parser.get<String>(0);
	String saliency_algorithm = "BinWangApr2014";
	String video_name = parser.get<String>(0);
	resizeMode = parser.get<int>(1);
	String training_path = parser.get<String>(2);
	//cout << parser.get<String>(0) << endl;

	String saliency_algorithm2 = "SPECTRAL_RESIDUAL";
	String saliency_algorithm3 = "FINE_GRAINED";
	//String saliency_algorithm4 = "BING"; //for objectness
	String face_cascade_name = "haar/haarcascade_frontalface_alt.xml";
	String body_cascade_name = "haar/haarcascade_mcs_upperbody.xml";

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!body_cascade.load(body_cascade_name)) { printf("--(!)Error loading body cascade\n"); return -1; };

	//Capture the video
	VideoCapture cap;

	if (video_name.length() == 1) {        //if the first argument have size 1
		cap.open(stoi(video_name));      // try to open camera device number (0,1...)
	}
	else {                                //else try to open a video file
		cout << video_name.length() << endl;
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
	Mat saliencyMap;


	//instantiates the specific Saliency
	Ptr<Saliency> saliencyAlgorithm = Saliency::create(saliency_algorithm);

	String choosenMethod = "";
	if (!method) { choosenMethod = saliency_algorithm2; }
	else { choosenMethod = saliency_algorithm3; }

	Ptr<Saliency> saliencyAlgorithm2 = Saliency::create(choosenMethod);

	//Ptr<Saliency> saliencyAlgorithm4 = Saliency::create(saliency_algorithm4); //for objectness

	if (saliencyAlgorithm == NULL)
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

	saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->setImagesize(image.cols, image.rows);
	saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->init();

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
				system("pause"); //will pause after program completion
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
			saliencyAlgorithm->computeSaliency(frame, saliencyMap);

			//Static saliency spectral residual or fine grained
			Mat saliencyMap2;
			saliencyAlgorithm2->computeSaliency(frame, saliencyMap2);

			//saliencyAlgorithm4->computeSaliency(image2, objVecMap); //for objectness

			//-- 3. Apply face detection to the frame
			detectFacesAndDisplay(frame, image);
			//-- 3. Apply contour detection to the frame
			detectContoursAndDisplay(saliencyMap2, image);

			//Let's compute percentage of moving pixels from the total frame pixels
			double whitePixelsCount = countNonZero(saliencyMap);
			double frameBlackPixelsPercent = (totalPixels - whitePixelsCount) / totalPixels;
			double frameWhitePixelsPercent = 1 - frameBlackPixelsPercent;

			//Estimation of motion difference between subsequent frames
			double motionFrameDiff = abs(frameWhitePixelsPercent - oldMotionValue);
			oldMotionValue = frameWhitePixelsPercent;
			if (motionFrameDiff > 0.05) { cuts++; }

			//Calculate mean and std. deviance for static and motion saliency
			Scalar tempVal = mean(saliencyMap2);
			double myMAtMean = tempVal.val[0];
			if (isnan(myMAtMean))myMAtMean = 0;
			runstatStaticSaliency.Push(myMAtMean);

			double s2 = runstatStaticSaliency.StandardDeviation();
			double m2 = runstatMotionSaliency.StandardDeviation();
			if (isnan(s2))s2 = 0;
			if (isnan(m2))m2 = 0;

			double facePercent = totalFaceBool / frameCount;
			//double bodyPercent = totalBodyBool / frameCount; //for body classification

			// Ad-hoc final metric
			finalAttentionScore = (runstatStaticSaliency.Mean() + s2
				+ runstatMotionSaliency.Mean() + m2 + (facePercent / 4) + (runstatContourArea.Mean() * 10) / totalPixels) / 3;
			String fas = "Score: " + std::to_string(finalAttentionScore);


			Point2f a(0, 0), b(140, 20);
			rectangle(image, a, b, (0, 0, 0), CV_FILLED, 8, 0);

			putText(image, fas, Point(5, 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1, 8, false);

			imshow("Original Image", image);
			//imshow("BING ojectness", image2);
			imshow("Motion Map", saliencyMap);
			imshow("Static map", saliencyMap2);


			cout << "\r" << std::setprecision(4) << std::fixed <<
				"fr: " << frameCount <<
				" SS: " << runstatStaticSaliency.Mean() <<
				" SS2: " << s2 <<
				" MS: " << runstatMotionSaliency.Mean() <<
				" MS2: " << m2 <<
				" F: " << facePercent <<
				" Cut: " << cuts <<
				" CA: " << runstatContourArea.Mean() * 10 / (totalPixels) <<
				" CN: " << runstatContours.Mean();

			//" MF: " << runstatMultipleFacesMean.Mean() <<

			if (frameCount > 80) runstatMotionSaliency.Push(frameWhitePixelsPercent);

			saliencyMap.release();
			saliencyMap2.release();
		}

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

	//}
	return 0;
}

/** @function detectFacesAndDisplay */
void detectFacesAndDisplay(const Mat& frame, const Mat& image)
{
	std::vector<Rect> faces;
	//std::vector<Rect> bodies;
	equalizeHist(frame, frame);

	//-- Detect faces
	face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, maxFaceSize);
	//body_cascade.detectMultiScale(frame, bodies, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, maxBodySize);

	if (faces.size())totalFaceBool++;

	if (drawFaces) {
		for (size_t i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(image, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 255, 255), 2, 8, 0);

		}
	}
	runstatMultipleFacesMean.Push(faces.size());
	/*
	if (bodies.size())totalBodyBool++;

	if (drawCascade2) {
		for (size_t i = 0; i < bodies.size(); i++)
		{
			Mat faceROI = image(bodies[i]);
			int x = bodies[i].x;
			int y = bodies[i].y;
			int h = y + bodies[i].height;
			int w = x + bodies[i].width;
			rectangle(image,
				Point(x, y),
				Point(w, h),
				Scalar(255, 0, 255),
				2,
				8,
				0);

		}
	}
	*/
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
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	double area0 = 0.0;
	/// Draw the contours
	if (drawTheContours)
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 0), rng.uniform(150, 255), rng.uniform(0, 255));
			drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
			vector<Point> contour = contours[i];
			area0 += contourArea(contour);
		}
	runstatContourArea.Push(area0);
	runstatContours.Push(contours.size());
}