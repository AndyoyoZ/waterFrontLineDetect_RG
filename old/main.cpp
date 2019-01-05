//#include "Scole.h"
#include "bankDetect.h"

int main()
{
	cv::Mat srcImage = cv::imread("D:/东方水利项目/data/image/1/image059.jpg",1);
	if (!srcImage.data)
	{
		std::cout << "Image Load error!" << std::endl;
		return 0;
	}
	cv::Mat resutImage;

	//FILELog::ReportingLevel() = logINFO;	//设置日志等级
	//IPSG::CScole Scole;
	//Scole.ModelsInital();
	//Scole.Process();

	////SLIC slic;
	////slic.slicSailency(srcImage, resutImage);

	IPSG::CbankDetect cbankdetect;
	if (cbankdetect.bankDetect(srcImage, resutImage,1))//岸体分割成功
	{
		cv::imshow("resutImage", resutImage);
		std::cout << "success" << std::endl;
	}
	else
		std::cout << "failed" << std::endl;

	cv::imshow("src", srcImage);
	
	cv::waitKey(0);
	cv::destroyAllWindows();//销毁所有highgui窗口
	system("pause");
	return EXIT_SUCCESS;
}