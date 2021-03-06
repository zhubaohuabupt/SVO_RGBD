// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <string.h>
#include <svo/global.h>
#include <svo/config.h>
#include <svo/frame.h>
#include <svo/feature_detection.h>
#include <svo/depth_filter.h>
#include <svo/feature.h>
#include <vikit/timer.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include "test_utils.h"
#include <svo/SGM.h>
namespace {

using namespace Eigen;
using namespace std;

void testCornerDetector()
{
  SGM* sgm;
  sgm = new SGM(80,1);
  char l_basename[25],r_basename[25];
  std::string img_path = "/home/bunny/SLAM/dataset/927/D3/";
  std::string l_img_name,r_img_name;
  for(int k=1746; k<3000; k+=50)
  {
	sprintf(l_basename, "01/left_%d.bmp", k);
        l_img_name = img_path + l_basename;
	
        sprintf(r_basename, "02/right_%d.bmp", k);
 	r_img_name = img_path + r_basename;
  	//printf("Loading image '%s'\n", img_name.c_str());
 	cv::Mat img_l(cv::imread(l_img_name, 0));//读左图
        cv::Mat img_r(cv::imread(r_img_name, 0));
  	assert(img_l.type() == CV_8UC1 && !img_l.empty());
  	vk::AbstractCamera* cam = new vk::ATANCamera(752, 480, 0.615919, 0.616737, 0.373040, 0.215405, 0);
  	svo::FramePtr frame(new svo::Frame(cam, img_l, 0.0));

  	// Corner detection
  	vk::Timer t;
  	svo::Features fts;
  	svo::feature_detection::FastDetector fast_detector(
      	img_l.cols, img_l.rows, svo::Config::gridSize(), svo::Config::nPyrLevels());
        /*******************************立体匹配**************************************************/
	cv::Mat A = Mat::zeros(img_l.rows, img_l.cols, CV_8UC1);
  	cv::Mat B = Mat::zeros(img_l.rows, img_l.cols, CV_8UC1);
  	cv::Mat left = img_l.clone();
  	cv::Mat right = img_r.clone(); 
  	sgm -> GetDisprity_mat(left,right,4,false,A,B);
  	/*****************************************************************************************/
  	for(int i=0; i<1; ++i)
  	{
    		fast_detector.detect(frame.get(), frame->img_pyr_, svo::Config::triangMinCornerScore(), fts);
  	}
  	//printf("Fast corner detection took %f ms, %zu corners detected (ref i7-W520: 7.166360ms, 40000)\n", t.stop()*10, fts.size());
  	//printf("Note, in this case, feature detection also contains the cam2world projection of the feature.\n");
  	//printf("第%d张图提取了%zu个特征点",k ,fts.size());
  	cv::Mat img_rgb = cv::Mat(img_l.size(), CV_8UC3);

  	cv::cvtColor(img_l, img_rgb, CV_GRAY2RGB);
  	double depth_temp;
  	double disparity_temp;
  	char d[15],f[15];
  	cv::Point point;
  	std::string text;
  	std::for_each(fts.begin(), fts.end(), [&](svo::Feature* i)
	{
		disparity_temp = A.at<uchar>(i->px[1],i->px[0]);
        	depth_temp = 0.06*615.919033/disparity_temp;
		/*for(int i=0;i<12;i++)
			d[i]='6';*/
		sprintf(d,"%f",depth_temp);
		//sprintf(f,"%f",depth_temp);
		//if(depth_temp)
		text = d;
        	point.x = i->px[0];
		point.y = i->px[1];
        	cv::putText(img_rgb,text,point,CV_FONT_HERSHEY_COMPLEX,0.4,Scalar(0,0,255));
        	cv::circle(img_rgb, cv::Point2f(i->px[0], i->px[1]), 1, cv::Scalar(0,255,0), 1);
    		//cv::circle(img_rgb, cv::Point2f(i->px[0], i->px[1]), 4*(i->level+1), cv::Scalar(0,255,0), 1);
		/*text = f;
        	cv::putText(img_rgb,text,point,CV_FONT_HERSHEY_COMPLEX,0.4,Scalar(0,0,255));
        	cv::circle(img_rgb, cv::Point2f(i->px[0], i->px[1]), 1, cv::Scalar(0,255,0), 1);*/
  	} );
  	cv::imshow("ref_img", img_rgb);
  	cv::waitKey(10);
	std::for_each(fts.begin(), fts.end(), [&](svo::Feature* i){ delete i; });//delet fts?
  }
  	
}

} // namespace


int main(int argc, char **argv)
{
  testCornerDetector();
  return 0;
}
