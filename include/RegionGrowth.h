#ifndef _REGION_GROETH_H
#define _REGION_GROETH_H
#include <list>
#include <opencv2/opencv.hpp>
#include <iostream>

//using namespace std;
//using namespace cv;

#define INIT 0    //未标记
#define SEED 100  //种子点
//#define CONT -2	 
#define INVAL 255 //边缘点

class g_point
{
public:
	g_point();
	g_point(int x_i, int y_i);
	g_point(int x_i, int y_i, int lbl_i);
public:
	int x;
	int y;
	int lbl;//标签


};

//构造函数
g_point::g_point()
{
	x = 0;
	y = 0;
	lbl = INIT;
}
g_point::g_point(int x_i, int y_i)
{
	x = x_i;
	y = y_i;
	lbl = SEED;
}

g_point::g_point(int x_i, int y_i, int lbl_i)
{
	x = x_i;
	y = y_i;
	lbl = lbl_i;
}


class RegionGrowth
{
private:
	
	void get_Seed_Point();
private:
	cv::Mat _src;
	cv::Mat _dst;
	typedef struct _Pix{
		int Pix_x;
		int Pix_y;
		float Pix_val;
		int Pix_num; 
	}_Pix_i;

	int _Seed_x;
	int _Seed_y;
	float _Seed_val;
	float _Pix_H;
	float _Pix_M;
	float _Thre;
public:
	RegionGrowth(cv::Mat& src, cv::Mat& dst, int x_g, int y_g, int thre);
};


RegionGrowth::RegionGrowth(cv::Mat& src, cv::Mat& dst, int x_g, int y_g, int thre):_Seed_x(x_g),_Seed_y(y_g),_Thre(thre)
{
	src.copyTo(_src);
	CV_Assert(_src.channels() == 1);//非灰度图像异常
	_src.convertTo(_src, CV_32S);
	dst.create(_src.rows, _src.cols, CV_32SC1);
	dst = cv::Scalar::all(0);
	int height = _src.rows;
	int width = _src.cols;
	int x, y;
	int gradient = 0;//梯度值

	//设置前次边缘点
	std::list<class g_point> cont_pts_pre;
	//std::list<class g_point> * pcont_pts_pre = &cont_pts_pre;

	//设置边缘点
	std::list<class g_point> cont_pts;
	//std::list<class g_point> * pcont_pts = &cont_pts;

	//初始种子点入队列
	cont_pts_pre.push_back( g_point(x_g, y_g/*, CONT*/));
	dst.ptr<int>(y_g)[x_g] = SEED;
	std::list<class g_point>::iterator iter_cont;
	//std::list<class g_point>::iterator iter_prt;
	std::list<class g_point>::iterator iter_swap;

	while (!cont_pts_pre.empty())
	{
		//一轮生长,采用四邻域处理
		iter_cont = cont_pts_pre.begin();
		while (iter_cont != cont_pts_pre.end())
		{
			x = (*iter_cont).x;
			y = (*iter_cont).y;
			///左
			if ((x - 1) >= 0)
			{
				if (dst.ptr<int>(y)[x - 1] == INIT)
				{
					gradient = _src.ptr<int>(y)[x] - _src.ptr<int>(y)[x - 1];
					if (abs(gradient) < thre) //满足阈值
					{
						cont_pts.push_back( g_point(x - 1, y/*, CONT*/));
						dst.ptr<int>(y)[x - 1] = SEED;
					}
					else//不满足阈值
					{
						dst.ptr<int>(y)[x - 1] = INVAL;
					}

				}
			}


			///上
			if ((y - 1) >= 0)
			{
				if (dst.ptr<int>(y - 1)[x] == INIT)
				{
					gradient = _src.ptr<int>(y)[x] - _src.ptr<int>(y - 1)[x];
					if (abs(gradient) < thre) //满足阈值
					{
						cont_pts.push_back( g_point(x, y - 1/*, CONT*/));
						dst.ptr<int>(y - 1)[x] = SEED;
					}
					else//不满足阈值
					{
						dst.ptr<int>(y - 1)[x] = INVAL;
					}
				}
			}

			///右
			if ((x + 1) < width)
			{
				if (dst.ptr<int>(y)[x + 1] == INIT)
				{
					gradient = _src.ptr<int>(y)[x] - _src.ptr<int>(y)[x + 1];
					if (abs(gradient) < thre) //满足阈值
					{
						cont_pts.push_back(g_point(x + 1, y/*, CONT*/));
						dst.ptr<int>(y)[x + 1] = SEED;
					}
					else//不满足阈值
					{
						dst.ptr<int>(y)[x + 1] = INVAL;
					}
				}
			}

			///下
			if ((y + 1) < height)
			{
				if (dst.ptr<int>(y + 1)[x] == INIT)
				{
					gradient = _src.ptr<int>(y)[x] - _src.ptr<int>(y + 1)[x];
					if (abs(gradient) < thre) //满足阈值
					{
						cont_pts.push_back( g_point(x, y + 1/*, CONT*/));
						dst.ptr<int>(y + 1)[x] = SEED;
					}
					else//不满足阈值
					{
						dst.ptr<int>(y + 1)[x] = INVAL;
					}
				}
			}
			iter_cont++;
		}
		// 将cont_pts中的点赋给cont_pts_pre
		cont_pts_pre.clear();
		iter_swap = cont_pts.begin();
		while (iter_swap != cont_pts.end())
		{
			cont_pts_pre.push_back(*iter_swap);
			iter_swap++;
		}
		cont_pts.clear();
	}
	cont_pts_pre.clear();
	cont_pts.clear();
}



#endif
