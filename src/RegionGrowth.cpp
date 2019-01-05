#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "RegionGrowth.h"

//#define RG_THRE   8.5    //��������������ֵ�趨������ֵΪ7,8,9,10

//using namespace std;
//using namespace cv;

int get_LNumPixVal(cv::Mat hist);
int get_RightContour(std::vector<std::vector<cv::Point> > Contours,int contourIdx1,int contourIdx2);
cv::Point get_ContourCenter(std::vector<cv::Point> contour);
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);
void BoundarySeedFill(cv::Mat &src, cv::Point ptStart);
void cal_mean_gradient(cv::Mat src);//����ƽ���ݶ�
float get_GRThre(cv::Mat lab_img);


int main(int argc, char* argv[])
{
	if(argc<2)
    {
        std::cout<<"usage:"<<std::endl;
        std::cout<<"      ./regionGrowth ../data/1.jpg"<<std::endl;
        return 0;
    }
    
    const char* image_path="";

    image_path=argv[1];

    cv::Mat srcImage;
    cv::Mat dstImage;
    

    srcImage = cv::imread(image_path);

    //�ж�ͼ���Ƿ��ȡ�ɹ�
    if(srcImage.empty())
    {
        std::cout << "image loading failed!" << std::endl;
        return -1;
    }
    else
        std::cout << "image loaded success!" << std::endl;
    
    //cv::resize(srcImage,srcImage,cv::Size(640,480));

	cv::Mat Lab_img;
	cv::cvtColor(srcImage,Lab_img,cv::COLOR_BGR2Lab);
    //cv::imshow("LAB",Lab_img);
//////////////////////�ָ��ߣ�otsu///////////////////////////////////////////////////////////
    //�ָ����ͨ��ͼ��
    std::vector<cv::Mat> channels;
    cv::split(Lab_img, channels);

    //cv::normalize(channels[1], channels[1], 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
    //cv::normalize(channels[2], channels[2], 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

    cv::imshow("L",channels[0]);
    cv::imshow("a",channels[1]);
    cv::imshow("b",channels[2]);

    //cv::merge(channels,Lab_img);
    //cv::imshow("merge",Lab_img);

    cv::Mat otsu_mask0,otsu_mask1,otsu_mask2;
    cv::threshold(channels[0], otsu_mask0, 0, 255, CV_THRESH_OTSU);
	cv::threshold(channels[1], otsu_mask1, 0, 255, CV_THRESH_OTSU);   
    cv::threshold(channels[2], otsu_mask2, 0, 255, CV_THRESH_OTSU);

    cv::imshow("otsu_mask0",otsu_mask0);
    cv::imshow("otsu_mask1",otsu_mask1);
    cv::imshow("otsu_mask2",otsu_mask2);

    cv::Mat thre_mask,dark_mask;
    //mask = otsu_mask0 &otsu_mask1+~otsu_mask2;
    thre_mask =otsu_mask0 & (otsu_mask1&~otsu_mask2);

    dark_mask=~otsu_mask0& ~otsu_mask1&otsu_mask2;
    cv::imshow("dark_mask",dark_mask);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	//������ 
	cv::morphologyEx(thre_mask, thre_mask, cv::MORPH_OPEN, element);
	//�ղ��� 
	cv::morphologyEx(thre_mask, thre_mask, cv::MORPH_CLOSE, element);
    cv::imshow("thre_mask",thre_mask);

    

/////////////////////////�ָ��ߣ���ȡ����///////////////////////////////////////////////////////////////////////
    std::vector<std::vector<cv::Point> > Contours;
	std::vector<cv::Vec4i> Hierarchy;
	cv::findContours(thre_mask, Contours, Hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::cout<<"Contour num:"<<Contours.size()<<std::endl;
	//�õ������������
	int maxContour1 = 0, maxContour2 = 0;
    cv::Point contour_Center;
    cv::Point Gravity_Center;
    cv::Mat draw_img;
    srcImage.copyTo(draw_img);

    if (!Contours.empty() && !Hierarchy.empty())
		{
			for (int i = 0; i < Contours.size(); i++)
			{
				if (/*Contours[i].size() > 50 && */Contours[i].size() > Contours[maxContour1].size())
				{
                    maxContour2=maxContour1;
					maxContour1 = i;
				}
			}
			//cv::drawContours(draw_img, Contours, maxContour1, cv::Scalar(255, 0, 0), 5);
            //cv::drawContours(draw_img, Contours, maxContour2, cv::Scalar(0, 255, 0), 5);
            //std::cout << "threshold success" << std::endl; 

            ///////////////////////////�ҵ���������ȡ����ͼ���·���һ������/////////////////////////////////////////////////////////////
            int RContourIdx;
            RContourIdx=get_RightContour(Contours,maxContour1,maxContour2);
            std::cout<<"maxContour1: "<<maxContour1<<std::endl;
            std::cout<<"maxContour2: "<<maxContour2<<std::endl;
            std::cout<<"RContourIdx: "<<RContourIdx<<std::endl;
            cv::drawContours(draw_img, Contours, RContourIdx, cv::Scalar(255, 0, 0), 5);
            //for(size_t i=0;i<Contours[RContourIdx].size();i++)
            //{
            //    std::cout<<"Contours[RContourIdx]"<<Contours[RContourIdx][i]<<std::endl;
            //} 
            ////////////////////////////�ָ��ߣ���ȡ��������/////////////////////////////////////////////////////////////////////////////         
            contour_Center=get_ContourCenter(Contours[RContourIdx]);
            std::cout<<"RcontourCenter:"<<contour_Center<<std::endl;
            cv::circle(draw_img,contour_Center,3,cv::Scalar(0,0,255),2);//��ɫ
            /*
            ////////////////////��������/�ָ��ߣ���ȡ��������///////////////////////////////////////////////////////////////
            cv::Moments m=cv::moments(Contours[RContourIdx],false);
            cv::Point2f Gravity_Center=cv::Point2f(m.m10/m.m00,m.m01/m.m00);        
            cv::circle(draw_img,Gravity_Center,3,cv::Scalar(0,255,0),2);
            std::cout<<"Gravity_Center:"<<Gravity_Center<<std::endl;
            */
		}
		else
		{
			std::cout << "threshold failed" << std::endl;	
		}

//////////////////////////�ָ��ߣ�����ֱ��ͼ����///////////////////////////////////////////////////////////////////
    //�趨bin��Ŀ
    int histBinNum = 256;

    //�趨ȡֵ��Χ
    float range[] = {0, 255};
    const float* histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    //��������ͨ����hist����
    cv::Mat hist_1, hist_2, hist_3;

    //����ֱ��ͼ
    cv::calcHist(&channels[0], 1, 0, cv::Mat(), hist_1, 1, &histBinNum, &histRange, uniform, accumulate);
    cv::calcHist(&channels[1], 1, 0, cv::Mat(), hist_2, 1, &histBinNum, &histRange, uniform, accumulate);
    cv::calcHist(&channels[2], 1, 0, cv::Mat(), hist_3, 1, &histBinNum, &histRange, uniform, accumulate);

	//�õ���Ŀ�������ص������ֵ
	int Pix_H[3];
	Pix_H[0]=get_LNumPixVal(hist_1);
	Pix_H[1]=get_LNumPixVal(hist_2);
	Pix_H[2]=get_LNumPixVal(hist_3);
	for(size_t i=0;i<3;i++)
	{
		std::cout<<"Pix_H["<<i<<"]="<<Pix_H[i]<<std::endl;
	}

    //����ֱ��ͼ����
    int hist_w = 400;
    int hist_h = 400;
    int bin_w = cvRound((double)srcImage.cols/histBinNum);

    cv::Mat histImage(srcImage.cols, srcImage.rows, CV_8UC3, cv::Scalar(0, 0, 0));
    //��ֱ��ͼ��һ������Χ[0, 255]
    cv::normalize(hist_1, hist_1, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist_2, hist_2, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist_3, hist_3, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());


    //ѭ������ֱ��ͼ
    for(int i = 1; i < histBinNum; i++)
    {
        cv::line(histImage, cv::Point(bin_w*(i-1), srcImage.rows - cvRound(hist_1.at<float>(i-1))),//cvRound��������
            cv::Point(bin_w*(i), srcImage.rows - cvRound(hist_1.at<float>(i))), cv::Scalar(0, 0, 255), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w*(i-1), srcImage.rows - cvRound(hist_2.at<float>(i-1))),
            cv::Point(bin_w*(i), srcImage.rows - cvRound(hist_2.at<float>(i))), cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w*(i-1), srcImage.rows - cvRound(hist_3.at<float>(i-1))),
            cv::Point(bin_w*(i), srcImage.rows - cvRound(hist_3.at<float>(i))), cv::Scalar(255, 0, 0), 2, 8, 0);
    }




    


/////////////////////////�ָ��ߣ���������/////////////////////////////////////////////////////////////////////////////////////////

    float RG_THRE=0;
    RG_THRE=get_GRThre(Lab_img);//�õ�����������Ҫ����ֵ

    cv::Mat gray_img,RG_img;
    cv::cvtColor(srcImage,gray_img,cv::COLOR_BGR2GRAY);
    cv::Point SeedPoint;
    int seed_X,seed_Y;
    SeedPoint=contour_Center;//�õ����ӵ㣬�˴���Ϊǰ���ȡ����������
    seed_X=SeedPoint.x;
    seed_Y=SeedPoint.y;

    RegionGrowth(gray_img,RG_img,seed_X,seed_Y,RG_THRE);//������������ʵ��
    RG_img.convertTo(RG_img, CV_8UC1);//��������ת��
	
    cv::Mat RG_mask;
    cv::threshold(RG_img, RG_mask, 0, 255, CV_THRESH_OTSU);
    //�ղ��� 
	cv::morphologyEx(RG_mask, RG_mask, cv::MORPH_CLOSE, element);
////////////////////////////////�ָ��ߣ��������������//////////////////////////////////////////////////////////////////////////////////////
    std::vector<std::vector<cv::Point> > RG_contours;
    cv::findContours(RG_mask,RG_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    cv::Mat fill_mask(thre_mask.size(),CV_8UC1,cv::Scalar(0));
    cv::drawContours(fill_mask, RG_contours, 0, cv::Scalar(255), 3);
    BoundarySeedFill(fill_mask,contour_Center);//���
    cv::imshow("fill_mask",fill_mask);

    cv::Mat RG_result;
    srcImage.copyTo(RG_result,fill_mask);
    cv::imshow("RG_result",RG_result);


    cv::imshow("gray",gray_img);
    cv::imshow("RG_img",RG_img);
    cv::imshow("draw_img",draw_img);
    
    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", srcImage);

    cv::namedWindow("hist", cv::WINDOW_AUTOSIZE);
    cv::imshow("hist", histImage);

    if(cv::waitKey(0)==27)
    {
        std::cout<<"normal exit!"<<std::endl;
    }
	
    return 0;
}


int get_LNumPixVal(cv::Mat hist)
{
	int LNum=0;
	int PixVal=0;
	int tempVal=0;
	int tempNum=0;
	for(size_t i=0;i<hist.rows;i++)
	{	
    	tempNum =cvRound(hist.at<float>(i));
		if(LNum<tempNum)
		{
			LNum=tempNum;
			tempVal=PixVal;
		}
		PixVal++;
	}
    
	//std::cout<<"Lnum "<<LNum<<std::endl;
	return tempVal;
}



void get_SeedPixVal(int* Pix_M,int*Pix_H)
{


}

/////////////////�����������л�ȡ����ͼ���·���һ������/////////////////////////////////////////////////
/////////////////������ȷ����������
int get_RightContour(std::vector<std::vector<cv::Point> > Contours,int contourIdx1,int contourIdx2)
{
    int32_t sum_y1=0,sum_y2=0;
    if(!Contours.empty())
    {
        for(size_t i=0;i<Contours[contourIdx1].size();i++)
        {
            sum_y1+=Contours[contourIdx1][i].y;
        }
        for(size_t i=0;i<Contours[contourIdx2].size();i++)
        {
            sum_y2+=Contours[contourIdx2][i].y;
        } 
        return sum_y1 > sum_y2 ? contourIdx1:contourIdx2;    
    }
    else
    {
        std::cout<<"error! contour size must == 2"<<std::endl;
        return 0;
    }

}


///////////�õ�����������
///////////������������

cv::Point get_ContourCenter(std::vector<cv::Point> contour)
{
    cv::Point center;
    int32_t sum_x=0,sum_y=0;
    int contourSize=contour.size();
    for(size_t i=0;i<contourSize;i++)
    {
        //std::cout<<"contour "<<contour<<std::endl;
        sum_x+=contour[i].x;
        sum_y+=contour[i].y;
    }
    center=cv::Point(sum_x/contourSize,sum_y/contourSize);
    if (cv::pointPolygonTest(contour,center,false) == 1)//�����������ڲ�
    {
        //std::cout<<"center "<<center<<std::endl;
        return center;
    }
    else
    {
        std::cout<<"error: center of gravity out of contour!"<<std::endl;
    }   
}



///////////////����������������ֵ��������
float get_GRThre(cv::Mat lab_img)
{
    float thre;
    cv::Mat means, stddev;
    cv::meanStdDev(lab_img, means, stddev);
    thre=stddev.at<double>(2,0);
    //std::cout<<"stddev:"<<stddev<<std::endl;
	//std::cout<<"thre:"<<thre<<std::endl;  
    return thre;  
}



bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2)
  {
    	if (mat1.empty() && mat2.empty()) {
    		return true;
    	}
    	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims||
    		mat1.channels()!=mat2.channels()) {
    		return false;
    	}
    	if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
    		return false;
    	}
    	int nrOfElements1 = mat1.total()*mat1.elemSize();
    	if (nrOfElements1 != mat2.total()*mat2.elemSize()) return false;
    	bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
    	return lvRet;
    }



/***************************************************
���ܣ���ptStartΪ����ͼ��������
������src-�߽�ͼ��
      ptStart-���ӵ������
****************************************************/
void BoundarySeedFill(cv::Mat &src, cv::Point ptStart)
{
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    uchar e[3][3] = {{ -1, 1, -1 }, { 1, 1, 1 }, { -1, 1, -1 }};//ʮ���νṹԪ��
    cv::Mat se(cv::Size(3, 3), CV_8UC1, e);
	cv::Mat tempImg = cv::Mat::ones(src.size(), src.type())*255;
	cv::Mat revImg =tempImg - src;//ԭͼ��Ĳ���
	dst.at<uchar>(ptStart.y, ptStart.x) = 255;//�������ӵ�
	while (true)//ѭ������ͼ��ֱ��ͼ���ڲ����仯
	{
		cv::Mat temp;
		dst.copyTo(temp);
		cv::dilate(dst, dst, se); //��ʮ�ֽṹԪ������
		dst = dst&revImg; //�������Ͳ��ᳬ��ԭʼ�߽�
		if (matIsEqual(dst, temp)) //���ڱ仯ʱֹͣ
		{
			break;
		}
	}
	src = dst;	
}



// ����ͼ���·��
// ����ͼ���ƽ���ݶ�
void cal_mean_gradient(cv::Mat src) 
{
    cv::Mat img;
    cv::cvtColor(src, img, cv::COLOR_BGR2GRAY); // ת��Ϊ�Ҷ�ͼ
    img.convertTo(img, CV_64FC1);
    double tmp = 0;
    int rows = img.rows - 1;
    int cols = img.cols - 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double dx = img.at<double>(i, j + 1) - img.at<double>(i, j);
            double dy = img.at<double>(i + 1, j) - img.at<double>(i, j);
            double ds = std::sqrt((dx*dx + dy*dy) / 2);
            tmp += ds;
        }
    }
    double imageAvG = tmp / (rows*cols);
    std::cout <<"gradient:" << imageAvG << std::endl;
}