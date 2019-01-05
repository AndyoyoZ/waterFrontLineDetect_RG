#ifndef BANKDETECT_H_
#define BANKDETECT_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <stdio.h>
#include <tchar.h>
#include <SDKDDKVer.h>

namespace IPSG
{
	class CbankDetect
	{
	public:
		CbankDetect(int _iLowL = 80, int _iHighL = 200, int _iLowA = 120, int _iHighA = 140, int _iLowB = 132, int _iHighB = 140):
					iLowL(_iLowL), iHighL(_iHighL), iLowA(_iLowA), iHighA(_iHighA), iLowB(_iLowB), iHighB(_iHighB)
		{}
		~CbankDetect()
		{}
		//-----------------bankDetect( )----------------------
		//----------------------------------------------------
		//                岸体检测与分割
		bool bankDetect(cv::Mat &inputImg, cv::Mat &outputImg,int thresholdMethod);



	private:
		//-----------------on_Trackbar( )-------------------
		//--------------------------------------------------
		//-------------------滑动条-------------------------
		void on_Trackbar(int, void*);

		//-----------------getPoint( )----------------------
		//--------------------------------------------------
		//            得到水岸线离散点集
		void getPoint(cv::Mat &img, std::vector<cv::Point> &inputPoint);

		//-----------------drawExtendLine( )----------------------
		//--------------------------------------------------------
		//                   画延长线
		void drawExtendLine(cv::Mat &img, cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4, cv::Scalar color, int thickness = 1, int line_type = 8);
		//-----------------segment( )----------------------
		//--------------------------------------------------
		//                  图像分割
		void segment(cv::Mat &inputImg, cv::Point pt1, cv::Point pt2, cv::Mat &outputImg);

		//distance -- max distance to the random line for voting
		//ngon     -- n-gon to be detected
		//itmax    -- max iteration times
		void ransacLines(std::vector<cv::Point>& input, std::vector<cv::Vec4d>& lines,
			double distance, unsigned int ngon, unsigned int itmax);

		//-----------------threshold_OTSU( )----------------------
		//----------------------------------------------------
		//                
		bool threshold_OTSU(cv::Mat &inputImg, cv::Mat &outputMask);

		//-----------------threshold_OTSU( )----------------------
		//----------------------------------------------------
		//                
		bool threshold_Lab(cv::Mat &inputImg, cv::Mat &outputMask);
	private:
		int iLowL;
		int iHighL;

		int iLowA;
		int iHighA;

		int iLowB;
		int iHighB;
	};
}

#endif