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

#include <svo/feature_detection.h>
#include <svo/feature.h>
#include <fast/fast.h>
#include <vikit/vision.h>
#include <opencv2/opencv.hpp>
#include<vector>
////////////zbh/////////////////////////
#include<cmath>
#include <vikit/params_helper.h>
extern bool Align_level_by_itsOwn_Point;//在frame_handler_mono.cpp定义
//////////////////////////////
using namespace std;
using namespace cv;
namespace svo {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{
////////////////////////////////zbh
display_Gradpiont=vk::getParam<bool>("svo/display_Gradpiont_");
//////////////////////////////////////////////////
}

void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    Features& fts)
{
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    vector<fast::fast_xy> fast_corners;
#if __SSE2__
      fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#elif HAVE_FAST_NEON
      fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#else
      fast::fast_corner_detect_10(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#endif
    vector<int> scores, nm_corners;
    fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);
////////////////////////zhb 测试特征点的分布对图像对齐精度的影响
# ifdef test1
for(int i=0;i<grid_n_rows_;i++)//图像右侧1/3不选特征点
for(int j=grid_n_cols_*2/3;j<grid_n_cols_;j++)
{
grid_occupancy_[i*grid_n_cols_+j]=1;
}
#endif
# ifdef test1
for(int i=0;i<grid_n_rows_/2;i++)//图像上半部分不选特征点
for(int j=0;j<grid_n_cols_;j++)
{
grid_occupancy_[i*grid_n_cols_+j]=1;
}
#endif
/////////////////////

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      fast::fast_xy& xy = fast_corners.at(*it);
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);
      if(grid_occupancy_[k])///////////////////////////////////////
        continue;
      const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      if(score > corners.at(k).score)
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
    }
  }

  // Create feature for every corner that has high enough corner score
  std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
    if(c.score > detection_threshold)
      fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
  });

  resetGrid();
}
////////////////////////////////////////////////////////////////////write by zhubaohua
void FastDetector::detectGrad(
    Frame* frame,
    const ImgPyr& img_pyr,
	 int max_level,
    Features& fts)
  {

vector <Grad> Grads;
  // for(int L=n_pyr_levels_-1; L<n_pyr_levels_; ++L)//n_pyr_levels_
      {
            int L=max_level;
          const int scale=(1<<L);
        Mat show=img_pyr[L].clone();
     cvtColor(show,show,CV_GRAY2RGB);
	const int this_level_width=img_pyr[L].cols;
	const int this_level_height=img_pyr[L].rows;
     Mat GX,GY,G;
	Sobel(img_pyr[L],GX,img_pyr[L].depth(),1,0);
	Sobel(img_pyr[L],GY,img_pyr[L].depth(),0,1);
		convertScaleAbs(GX, GX);//////////////
		convertScaleAbs(GY, GY);
		addWeighted(GX, 0.5, GY, 0.5, 0, G);
		for(int row=0;row<this_level_height;row++)
		   for(int col=0;col<this_level_width;col++)
		{
		       // if(G.at<uchar>(row,col)>detection_threshold)
                       //最内层全提
				{
				    Grad grad_tmp(col*scale,row*scale,L);
				    Grads.push_back(grad_tmp);
                    circle(show,Point2f(col,row),2,Scalar(0,0,255));   
                                   
				}
                   
               }



//#define imshow
#ifdef imshow

{
imshow("grad",show);
waitKey(0);
}
#endif		
       }


//把Grads里的梯度点放到输出接口fts里
		
                      for(int i=0;i<Grads.size();i++)
                          {
                            fts.push_back(new Feature(frame, Vector2d(Grads[i].x, Grads[i].y), Grads[i].level));//梯度点   
                         }
   }

////////////////////////////////////////////////////////////////////write by zhubaohua///////////////
int FastDetector::detectGrad_cell( Frame* frame,const ImgPyr& img_pyr, 
  			int level0_gridsize,int want_piont_num,int Constant_thred,
			Features& fts,float gridsize_down_ratio)
{
if(display_Gradpiont)
cout<<"开始对新的一帧用网格法提取梯度点"<<endl;
bool stop=false;
int constant_thred=Constant_thred;
 int get_piont_num=0;
 int last_get_num=0;//用来保存上一轮提取的梯度点数
vector <Grad> Grads;
int cur_level_gridsize=level0_gridsize;
for(int L=0;L<n_pyr_levels_;++L)//n_pyr_levels_
   {  
                       int get_piont_num_each_level=0;//用来统计每层提取点的数量
				 //显示用
				  Mat testshow=img_pyr[L].clone();
                             cvtColor(testshow,testshow,CV_GRAY2BGR);
				
	 const int scale =1<<L;
	 //计算梯度
	 Mat GX,GY,G;
	Sobel(img_pyr[L],GX,img_pyr[L].depth(),1,0);
	Sobel(img_pyr[L],GY,img_pyr[L].depth(),0,1);
	convertScaleAbs(GX,GX);
	convertScaleAbs(GY,GY);
	addWeighted(GX,0.5,GY,0.5,0,G);
	//imshow("",G);
	//waitKey(0);
	 ///
  const int this_level_width=img_pyr[L].cols;
  const int this_level_height=img_pyr[L].rows;
 cur_level_gridsize=level0_gridsize/pow(gridsize_down_ratio,L);
  int cell_h_num=this_level_height/cur_level_gridsize;
  int cell_w_num=this_level_width/cur_level_gridsize;
/////////////////////////////////最内层不用网格法提取

 
   if(L==n_pyr_levels_-1)  
     {  
      
           int Point_cnt= Grads.size();
    for(int row=0;row<this_level_height;row++)
       for(int col=0;col<this_level_width;col++)
	 {
	  //if(G.at<uchar>(row,col)>10)
               {      
                   Grad large_grad(col,row,L);//构造
              //转化到原尺度图像，是最终提取的点。L标志金字塔层数来源
            Grad maxgrad_pointTolevel0(large_grad.x*scale,large_grad.y*scale,L);
		   Grads.push_back(maxgrad_pointTolevel0);
              if(display_Gradpiont)
              circle(testshow,Point2f(large_grad.x,large_grad.y),0.2,Scalar(0,0,255));
              }
                  
         }
			if(display_Gradpiont)
			{
                         cout<<"---------------->>第 "<<L<<"层共提取了"<<Grads.size()-Point_cnt<<" 个梯度点"<<endl;
			imshow("按方格提取",testshow);
			waitKey(0);
			}
       goto breakLoop_Pyramid;
   }

//////////////////////////////////3-0层用网格法提取////////////////////////////////

 
   
  vector<bool> ischoose(this_level_width*this_level_height,false);
//选点
for(int ite_getpiont=1;ite_getpiont<3;ite_getpiont++)
    {

  cell_h_num=cell_h_num/ite_getpiont;
  cell_w_num=cell_w_num/ite_getpiont;

   cur_level_gridsize= cur_level_gridsize*ite_getpiont;
	vector<int> Dynamic_grad_thred(cell_h_num*cell_w_num);
	for(int j=0;j<cell_h_num;j++)
	for(int i=0;i<cell_w_num;i++)
         {
      
      //计算每个窗口内的动态阈值
        double sumsobel=0;
	    for(int row=j*cur_level_gridsize;row<j*cur_level_gridsize+cur_level_gridsize;row++)
           for(int col=i*cur_level_gridsize;col<i*cur_level_gridsize+cur_level_gridsize;col++)
             		{
			sumsobel+=G.at<uchar>(row,col);
		       }
        int cell_point_num= cur_level_gridsize* cur_level_gridsize;
         Dynamic_grad_thred.push_back(sumsobel/cell_point_num+constant_thred);//计算每个方格的平均值
     
          Grad maxgrad(0,0,0);
           for(int row=j* cur_level_gridsize;row<j* cur_level_gridsize+cur_level_gridsize;row++)
            for(int col=i* cur_level_gridsize;col<i* cur_level_gridsize+cur_level_gridsize;col++)
              {
                  
			 if(G.at<uchar>(row,col)> Dynamic_grad_thred.back())
                         
                              {
                               if(ischoose[row*this_level_width+col]==true)//前面选过的点不要
								continue;
				if(G.at<uchar>(row,col)>G.at<uchar>(maxgrad.y,maxgrad.x))//选此方格内的梯度最大点
                      	    		     {
							Grad tmp(col,row,L);
							 maxgrad=tmp;
					     }
 			    }  
                         
            }
           if((maxgrad.x==0)&&(maxgrad.y==0)&&(maxgrad.level==0))
                      continue;
       Grad maxgrad_pointTolevel0(maxgrad.x*scale,maxgrad.y*scale,L);//转化到原尺度图像，是最终提取的点。L标志金字塔层数来源
		   Grads.push_back(maxgrad_pointTolevel0);
	if(display_Gradpiont)
 	circle(testshow,Point2f(maxgrad.x,maxgrad.y),2,Scalar(0,0,255));

           ++get_piont_num;  //统计总共提取的点数
          ++get_piont_num_each_level;//统计每层金字塔提取的点数
//选过点的周围 window_size*window_size 不让再选点，即以该点为中心的window_size*window_size 窗口内全置true ，不让再选点
          int window_size=3;
          for(int j=- window_size/2;j!= window_size/2+1;j++)
	    for(int i= window_size/2;i!= window_size/2+1;i++)
		  {
 	    	   int window_row=maxgrad.y+j;
		   int window_col=maxgrad.x+i;
		if( window_row>=0&& window_row<this_level_height&& window_col>=0&& window_col<this_level_width)
        	  ischoose[this_level_width*window_row+window_col]=true;
		  }
		  if (get_piont_num>=want_piont_num)
		   {
                        //设置总共提取点的最大数量，超过这个值，结束提取程序。
                            // if(!Align_level_by_itsOwn_Point)
		              //  { 
                             //      stop=true;
		              //   goto breakLoop_cells;
                                 //}
		   }
		
			 
         }

breakLoop_cells:      
         //输出每轮提取的梯度点数
	  if(display_Gradpiont)
		{
	  	 cout<<"第 "<<L<<"层 第"<<ite_getpiont<<" 轮提取了 "<<get_piont_num-last_get_num<<endl;
 		}  
              last_get_num=get_piont_num;   
         if(get_piont_num_each_level>want_piont_num) break;//每层经过某一轮提取的梯度点数一旦超过want_piont_num，停止本层提取。
                                              
                           if(stop==true)
                              goto breakLoop_change_cell_size;
        }//end...for(int ite_getpiont=1;ite_getpiont<3;ite_getpiont++)
              if(display_Gradpiont)
              cout<<"---------------->>第 "<<L<<"层共提取了"<<get_piont_num_each_level<<" 个点 "<<endl;
  
breakLoop_change_cell_size:
		    if(display_Gradpiont)
			{
			imshow("按方格提取",testshow);
			waitKey(0);
			}
                           if(stop==true)
                             goto breakLoop_Pyramid;
   
 }//end....for(int L=0;L<n_pyr_levels_;++L)
breakLoop_Pyramid:
                   
                    if(display_Gradpiont)
			{
                        cout<<"按论文中的方法目前共提取了 "<<Grads.size()<<"个点"<<"  效果如图"<<endl;
                     	}
                   for(int i=0;i<Grads.size();i++)
                          {
                            fts.push_back(new Feature(frame, Vector2d(Grads[i].x, Grads[i].y), Grads[i].level));//梯度点   
                         }
                   return Grads.size();
 
}
int FastDetector::detectGrad_cell_level0_2( Frame* frame,const ImgPyr& img_pyr, 
  			int level0_gridsize,int want_piont_num,int Constant_thred,
			Features& fts,float gridsize_down_ratio)
{
if(display_Gradpiont)
cout<<"开始对新的一帧用网格法提取梯度点"<<endl;
bool stop=false;
int constant_thred=Constant_thred;
 int get_piont_num=0;
 int last_get_num=0;//用来保存上一轮提取的梯度点数
vector <Grad> Grads;
int cur_level_gridsize=level0_gridsize;
for(int L=0;L<3;++L)//n_pyr_levels_  //只提取level0-level2上的梯度点
   {  
                       int get_piont_num_each_level=0;//用来统计每层提取点的数量
				 //显示用
				  Mat testshow=img_pyr[L].clone();
                             cvtColor(testshow,testshow,CV_GRAY2BGR);
				
	 const int scale =1<<L;
	 //计算梯度
	 Mat GX,GY,G;
	Sobel(img_pyr[L],GX,img_pyr[L].depth(),1,0);
	Sobel(img_pyr[L],GY,img_pyr[L].depth(),0,1);
	convertScaleAbs(GX,GX);
	convertScaleAbs(GY,GY);
	addWeighted(GX,0.5,GY,0.5,0,G);
	//imshow("",G);
	//waitKey(0);
	 ///
  const int this_level_width=img_pyr[L].cols;
  const int this_level_height=img_pyr[L].rows;
 cur_level_gridsize=level0_gridsize/pow(gridsize_down_ratio,L);
  int cell_h_num=this_level_height/cur_level_gridsize;
  int cell_w_num=this_level_width/cur_level_gridsize;
   
  vector<bool> ischoose(this_level_width*this_level_height,false);
//选点
for(int ite_getpiont=1;ite_getpiont<3;ite_getpiont++)
    {

  cell_h_num=cell_h_num/ite_getpiont;
  cell_w_num=cell_w_num/ite_getpiont;

   cur_level_gridsize= cur_level_gridsize*ite_getpiont;
	vector<int> Dynamic_grad_thred(cell_h_num*cell_w_num);
	for(int j=0;j<cell_h_num;j++)
	for(int i=0;i<cell_w_num;i++)
         {
      
      //计算每个窗口内的动态阈值
        double sumsobel=0;
	    for(int row=j*cur_level_gridsize;row<j*cur_level_gridsize+cur_level_gridsize;row++)
           for(int col=i*cur_level_gridsize;col<i*cur_level_gridsize+cur_level_gridsize;col++)
             		{
			sumsobel+=G.at<uchar>(row,col);
		       }
        int cell_point_num= cur_level_gridsize* cur_level_gridsize;
         Dynamic_grad_thred.push_back(sumsobel/cell_point_num+constant_thred);//计算每个方格的平均值
     
          Grad maxgrad(0,0,0);
           for(int row=j* cur_level_gridsize;row<j* cur_level_gridsize+cur_level_gridsize;row++)
            for(int col=i* cur_level_gridsize;col<i* cur_level_gridsize+cur_level_gridsize;col++)
              {
                  
			 if(G.at<uchar>(row,col)> Dynamic_grad_thred.back())
                         
                              {
                               if(ischoose[row*this_level_width+col]==true)//前面选过的点不要
								continue;
				if(G.at<uchar>(row,col)>G.at<uchar>(maxgrad.y,maxgrad.x))//选此方格内的梯度最大点
                      	    		     {
							Grad tmp(col,row,L);
							 maxgrad=tmp;
					     }
 			    }  
                         
            }
           if((maxgrad.x==0)&&(maxgrad.y==0)&&(maxgrad.level==0))
                      continue;
       Grad maxgrad_pointTolevel0(maxgrad.x*scale,maxgrad.y*scale,L);//转化到原尺度图像，是最终提取的点。L标志金字塔层数来源
		   Grads.push_back(maxgrad_pointTolevel0);
	if(display_Gradpiont)
 	circle(testshow,Point2f(maxgrad.x,maxgrad.y),2,Scalar(0,0,255));

           ++get_piont_num;  //统计总共提取的点数
          ++get_piont_num_each_level;//统计每层金字塔提取的点数
//选过点的周围 window_size*window_size 不让再选点，即以该点为中心的window_size*window_size 窗口内全置true ，不让再选点
          int window_size=3;
          for(int j=- window_size/2;j!= window_size/2+1;j++)
	    for(int i= window_size/2;i!= window_size/2+1;i++)
		  {
 	    	   int window_row=maxgrad.y+j;
		   int window_col=maxgrad.x+i;
		if( window_row>=0&& window_row<this_level_height&& window_col>=0&& window_col<this_level_width)
        	  ischoose[this_level_width*window_row+window_col]=true;
		  }
		  if (get_piont_num>=want_piont_num)
		   {
                        //设置总共提取点的最大数量，超过这个值，结束提取程序。
                            // if(!Align_level_by_itsOwn_Point)
		              //  { 
                             //      stop=true;
		              //   goto breakLoop_cells;
                                 //}
		   }
		
			 
         }

breakLoop_cells:      
         //输出每轮提取的梯度点数
	  if(display_Gradpiont)
		{
	  	 cout<<"第 "<<L<<"层 第"<<ite_getpiont<<" 轮提取了 "<<get_piont_num-last_get_num<<endl;
 		}  
              last_get_num=get_piont_num;   
         if(get_piont_num_each_level>want_piont_num) break;//每层经过某一轮提取的梯度点数一旦超过want_piont_num，停止本层提取。
                                              
                           if(stop==true)
                              goto breakLoop_change_cell_size;
        }//end...for(int ite_getpiont=1;ite_getpiont<3;ite_getpiont++)
              if(display_Gradpiont)
              cout<<"---------------->>第 "<<L<<"层共提取了"<<get_piont_num_each_level<<" 个点 "<<endl;
  
breakLoop_change_cell_size:
		    if(display_Gradpiont)
			{
			imshow("按方格提取",testshow);
			waitKey(0);
			}
                           if(stop==true)
                             goto breakLoop_Pyramid;
   
 }//end....for(int L=0;L<n_pyr_levels_;++L)
breakLoop_Pyramid:
                   
                    if(display_Gradpiont)
			{
                        cout<<"按论文中的方法目前共提取了 "<<Grads.size()<<"个点"<<"  效果如图"<<endl;
                     	}
                   for(int i=0;i<Grads.size();i++)
                          {
                            fts.push_back(new Feature(frame, Vector2d(Grads[i].x, Grads[i].y), Grads[i].level));//梯度点   
                         }
                   return Grads.size();
 
}
/////////////////////////////////////////////////////////write by zhubaohau/////////////////////////////////////////

} // namespace feature_detection
} // namespace svo

