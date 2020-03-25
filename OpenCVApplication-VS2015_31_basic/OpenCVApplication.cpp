// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "stdafx.h"
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
using namespace cv;



void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



Mat canny(Mat src) {
	Mat Gx = Mat(src.rows, src.cols, CV_32FC1);
	Mat Gy = Mat(src.rows, src.cols, CV_32FC1);
	Mat G = Mat(src.rows, src.cols, CV_32FC1);
	Mat GN = Mat(src.rows, src.cols, CV_32FC1);
	Mat fi = Mat(src.rows, src.cols, CV_32FC1);
	Mat dstnou = Mat(src.rows, src.cols, CV_32FC1);
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	//Mat Matx, Maty;

	int Matx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	int Maty[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };


	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			float convx = 0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					convx += Matx[k][l] * src.at<uchar>(i + k - 1, j + l - 1);//v.push_back(src.at<uchar>(i + k, j + m));
					//conv += aux.at<float>(k + dim / 2, m + dim / 2) * src.at<uchar>(i + k, j + m);
				}
			}
			Gx.at<float>(i, j) = convx;
		}
	}
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			float convy = 0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					convy += Maty[k][l] * src.at<uchar>(i + k - 1, j + l - 1); //v.push_back(src.at<uchar>(i + k, j + m));
				}
			}
			Gy.at<float>(i, j) = convy;
		}
	}

	for (int i = 0; i < G.rows; i++)
	{
		for (int j = 0; j < G.cols; j++)
		{
			G.at<float>(i, j) = sqrt(pow(Gx.at<float>(i, j), 2) + pow(Gy.at<float>(i, j), 2));
			fi.at<float>(i, j) = atan2(Gy.at<float>(i, j), Gx.at<float>(i, j));
		}
	}

	for (int i = 0; i < G.rows; i++)
	{
		for (int j = 0; j < G.cols; j++)
		{
			GN.at<float>(i, j) = G.at<float>(i, j) / (4 * sqrt(2));
		}
	}



	for (int i = 1; i < fi.rows - 1; i++)
	{
		for (int j = 1; j < fi.cols - 1; j++)
		{
			//0
			if ((fi.at<float>(i, j) >= (-PI / 8) && fi.at<float>(i, j) <= (PI / 8)) || fi.at<float>(i, j) <= (-7 * PI / 8) || fi.at<float>(i, j) > (7 * PI / 8))
			{
				if (GN.at<float>(i, j) > GN.at<float>(i, j - 1) && GN.at<float>(i, j) > GN.at<float>(i, j + 1))
				{
					dstnou.at<float>(i, j) = GN.at<float>(i, j);
				}
				else
					dstnou.at<float>(i, j) = 0;
			}
			//1
			if ((fi.at<float>(i, j) > (PI / 8) && fi.at<float>(i, j) <= (3 * PI / 8)) || (fi.at<float>(i, j) <= (-5 * PI / 8) && fi.at<float>(i, j) > (-7 * PI / 8)))
			{
				if (GN.at<float>(i, j) > GN.at<float>(i - 1, j + 1) && GN.at<float>(i, j) > GN.at<float>(i + 1, j - 1))
				{
					dstnou.at<float>(i, j) = GN.at<float>(i, j);
				}
				else
					dstnou.at<float>(i, j) = 0;
			}
			//2
			if ((fi.at<float>(i, j) > (3 * PI / 8) && fi.at<float>(i, j) <= (5 * PI / 8)) || (fi.at<float>(i, j) <= (-3 * PI / 8) && fi.at<float>(i, j) > (-5 * PI / 8)))
			{
				if (GN.at<float>(i, j) > GN.at<float>(i - 1, j) && GN.at<float>(i, j) > GN.at<float>(i + 1, j))
				{
					dstnou.at<float>(i, j) = GN.at<float>(i, j);
				}
				else
					dstnou.at<float>(i, j) = 0;
			}

			//3
			if ((fi.at<float>(i, j) > (5 * PI / 8) && fi.at<float>(i, j) <= (7 * PI / 8)) || (fi.at<float>(i, j) < (-PI / 8) && fi.at<float>(i, j) > (-3 * PI / 8)))
			{
				if (GN.at<float>(i, j) > GN.at<float>(i - 1, j - 1) && GN.at<float>(i, j) > GN.at<float>(i + 1, j + 1))
				{
					dstnou.at<float>(i, j) = GN.at<float>(i, j);
				}
				else
					dstnou.at<float>(i, j) = 0;
			}
		}
	}

	for (int i = 0; i < G.rows; i++)
	{
		for (int j = 0; j < G.cols; j++)
		{
			if (dstnou.at<float>(i, j) < 23) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
			//dst.at<uchar>(i, j) = dstnou.at<float>(i, j);
		}
	}

	return dst;
}

//Chamfer DT, 3x3 mask splitted in 2
Mat distanceTransform(Mat srcImg) {
	Mat DT = srcImg.clone();
	int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	int weights[8] = { 3,2,3, 2, 2, 3, 2, 3 };
	int min = 0, currentValue = 0;
	//first pass
	for (int i = 1; i < srcImg.rows - 1; i++) {
		for (int j = 1; j < srcImg.cols - 1; j++) {
			min = INT_MAX;
			//use first part of the mask
			for (int k = 0; k < 4; k++) {
				currentValue = DT.at<uchar>(i + di[k], j + dj[k]) + weights[k];
				if (currentValue < min) {
					min = currentValue;
				}
			}
			if (min < DT.at<uchar>(i, j))
				DT.at<uchar>(i, j) = min;
		}
	}
	//second pass
	for (int i = srcImg.rows - 2; i > 0; i--) {
		for (int j = srcImg.cols - 2; j > 0; j--) {
			min = INT_MAX;
			for (int k = 4; k < 8; k++) {
				currentValue = DT.at<uchar>(i + di[k], j + dj[k]) + weights[k];
				if (currentValue < min) {
					min = currentValue;
				}
			}
			if (min < DT.at<uchar>(i, j))
				DT.at<uchar>(i, j) = min;
		}

	}
	return DT;
}

uchar invert(uchar color) {
	if (color < 127) {
		return 255;
	}
	else {
		return 0;
	}
}

void detectiaCanny(Mat model) {

	Mat src;
	char fname[MAX_PATH];
	int aux1;


	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat cm = canny(src);
		int modelSize = 0;
		for (int i = 0; i < model.rows; i++) {
			for (int j = 0; j < model.cols; j++) {
				if (model.at<uchar>(i, j) == 0) {
					modelSize++;
				}
			}
		}
		double bestScore = 10000.0;
		int bestI=0, bestJ=0;
		for (int i = 0; i < cm.rows - model.rows; i++) {
			if(i%20 == 0) printf("\r %d / %d", i, cm.rows - model.rows);
			for (int j = 0; j < cm.cols - model.cols; j++) {
				double contourPixels = 0.0;
				double score = 0.0;
				for (int i1 = 0; i1 < model.rows; i1++) {
					for (int j1 = 0; j1 < model.cols; j1++) {
						if (cm.at<uchar>(i+i1, j+j1) == 0)
						{
							score += (double)model.at<uchar>(i1, j1);
							contourPixels += 1.0;
						}
					}
				}
				if (contourPixels > modelSize / 2) {
					score /= contourPixels;
					if (score < bestScore) {
						bestScore = score;
						bestI = i;
						bestJ = j;
					}
				}
			}
		}
		printf("\nbestScore=%lf; bestI=%d; bestJ=%d\n", bestScore, bestI, bestJ);
		Mat dst = src.clone();
		for (int i = bestI; i < bestI + model.rows; i++) {
			dst.at<uchar>(i, bestJ) = invert(dst.at<uchar>(i, bestJ));
			dst.at<uchar>(i, bestJ+model.cols) = invert(dst.at<uchar>(i, bestJ + model.cols));
		}
		for (int j = bestJ; j < bestJ + model.cols; j++) {
			dst.at<uchar>(bestI, j) = invert(dst.at<uchar>(bestI, j));
			dst.at<uchar>(bestI + model.rows, j) = invert(dst.at<uchar>(bestI + model.rows, j));
		}
		imshow("canny", cm);
		imshow("result", dst);
		waitKey();

	}


}

void detectieCutie() {
	Mat imagine = imread("Images\\Cutie2.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat binar1 = canny(imagine);
	Mat sablon = distanceTransform(binar1);
	imshow("binar1", binar1);
	imshow("sablon", sablon);
	detectiaCanny(sablon);
	waitKey();

}



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Proiect\n\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				detectieCutie();
				break;
		}
	}
	while (op!=0);
	return 0;
}