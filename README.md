# Attention score 

Experimental utility for video attention score metric using a fusion of visual attention related features. Based on the [(OpenCV library of algorithms](https://opencv.org/). This code is just a proof of concept.

![figure 1](/images/screenshot.jpg)
*figure 1 - Application Interface* 

## Installation

* There is a Visual Studio 2015 project
* OpenCV 3.2.0 is required, including OPENCV_DIR Path environment variable correctly set.
* or simply copy the bin/ folder to any windows x64 machine and run attentionScore executable providing proper arguments. Also provided are the openCV related DLLs to be possible to run the application in a system without openCV installed or correct Path environment variable setup (e.g: OPENCV_DIR).

## Possible inputs

The program captures frames from a video file, or camera connected to your computer.

To capture from a camera pass the device number (e.g: 0, 1).

* attentionScore 0, will result in capturing the camera device 0.

* You may also pass a video file instead of a device number, as in attentionScore movies\video.avi

* To open preview windows use "1" as second argument (attentionScore movies\video.mp4 1 or attentionScore 0 1)

* To process a folder(default: .\movies) of video files use the available batch file (process_folder.bat)

## Output

![figure 2](/images/output.jpg)
*figure 2 - Output*

The metrics and final attention score for each processed video is appended in results\score_output.csv

The content of score_output.csv and metrics related to each column:

movies\1_fingers.mov,0.175943,0.473721,0.00460829,0,0.300968,3.56221,0.406569,0.29589
movies\5_mov_480x360.mov,0.315486,0.0977334,0.99537,0,0.299118,10.1204,0.429833,0.331092
movies\6_pendulum_420.mov,0.114615,0.107616,0.00429185,0,0.207318,3.31116,0.277322,0.148601
movies\animados.mov,0.329181,0.560134,0.134021,5,0.517437,7.07474,0.5689,0.42837
movies\Black_dot.mov,0.117989,0.00859495,0,0,0.239875,1.64631,0,0.0621175
movies\red_circle2.mov,0.0747458,0.0189823,0,0,0.130848,3.57358,0.214174,0.0910011

* video file path
* Static saliency mean
* Motion saliency mean
* Face detection percent
* Cuts detected by motion saliency comparison
* Contour area, obtained from the static saliency map using a treshold
* Contour count, mean count of individual contours per frame
* Colourfulness
* Final score

## Keyboard shortcuts for video preview

* q : quit
* p : pause
* 1 : toggle face detection
* 2 : toggle contours
* 3 : reset values, usefull for camera device

## Additional info

* [OpenCV Saliency API](https://docs.opencv.org/3.0-beta/modules/saliency/doc/saliency.html)

Many computer vision applications may benefit from understanding where humans focus given a scene. Other than cognitively understanding the way human perceive images and scenes, finding salient regions and objects in the images helps various tasks such as speeding up object detection, object recognition, object tracking and content-aware image editing.

* [Saliency Detection: A Spectral Residual Approach](https://www.google.pt/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjnxZbw8qXaAhUD0RQKHToFDPwQFggsMAA&url=http%3A%2F%2Fbcmi.sjtu.edu.cn%2F~zhangliqing%2FPapers%2F2007CVPR_Houxiaodi_04270292.pdf&usg=AOvVaw2ofGQaPXbfGjDvt3mnsILR)

Starting from the principle of natural image statistics, this method simulate the behavior of pre-attentive visual search. The algorithm analyze the log spectrum of each image and obtain the spectral residual. Then transform the spectral residual to spatial domain to obtain the saliency map, which suggests the positions of proto-objects.

* [A Fast Self-tuning Background Subtraction Algorithm](https://www.google.pt/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjTu_fm86XaAhXCSBQKHasaA5QQFggsMAA&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F9752%2F9871deda00fea80a9781e29189970553812e.pdf&usg=AOvVaw0GCrl171KenpGNEVtima1z)

Algorithms belonging to this category, are particularly focused to detect salient objects over time (hence also over frame), then there is a temporal component sealing cosider that allows to detect “moving” objects as salient, meaning therefore also the more general sense of detection the changes in the scene.

* [findContours](https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html)

Finding contours in a image

* [Cascade classifiers](https://docs.opencv.org/3.0-beta/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html)

Use the CascadeClassifier class to detect objects in a video stream

* [RunningStats](https://www.johndcook.com/blog/skewness_kurtosis/)

The code is an extension of the method of Knuth and Welford for computing standard deviation in one pass through the data. It computes skewness and kurtosis as well with a similar interface. In addition to only requiring one pass through the data, the algorithm is numerically stable and accurate


## Also experimenting

[Objectness](https://docs.opencv.org/3.0-beta/modules/saliency/doc/objectness_algorithms.html)

Objectness is usually represented as a value which reflects how likely an image window covers an object of any category. Algorithms belonging to this category, avoid making decisions early on, by proposing a small number of category-independent proposals, that are expected to cover all objects in an image. Being able to perceive objects before identifying them is closely related to bottom up visual attention (saliency)