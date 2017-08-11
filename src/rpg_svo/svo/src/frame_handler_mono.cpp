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

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <cxcore.hpp>
#include <cv.h>
#include <highgui.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif
#include<iostream>
/////////////////////////////zhubaohua
#include <fstream>
#include <vikit/params_helper.h>
ofstream svo_inf("/home/baohua/SLAM/SVO_RGBD_grad/catkin_ws/src/rpg_svo/svo/src/T/svo_inf.txt",ios::out);//
ofstream ALIGN("/home/baohua/SLAM/SVO_RGBD_grad/catkin_ws/src/rpg_svo/svo/src/T/ALIGN.txt",ios::out);//
ofstream max_level_tran("/home/baohua/SLAM/SVO_RGBD_grad/catkin_ws/src/rpg_svo/svo/src/T/max_level_tran.txt");
ofstream imu_detaT("/home/baohua/SLAM/SVO_RGBD_grad/catkin_ws/src/rpg_svo/svo/src/T/imu_detaT.txt");
ofstream detaT("/home/baohua/SLAM/SVO_RGBD_grad/catkin_ws/src/rpg_svo/svo/src/T/detaT.txt");
ofstream detaT_ORB("/home/baohua/SLAM/SVO_RGBD_grad/catkin_ws/src/rpg_svo/svo/src/T/detaT_ORB.txt");
//#define cam_6cm
#define MH01

static int keyf=0;
static int frame=1;
//此两个参数是修改金字塔相关的参数
float benmark_pyr_ratio;
bool benmark_change_pyr_ratio;
//此参数是修改对齐点，让每层对齐只用本层的特征点。
bool Align_level_by_itsOwn_Point;//在sparse_img_align.cpp里用到。
bool FastConer0_2_Grad4;//在sparse_img_align.cpp里用到。
bool show_message;//在sparse_img_align.cpp里用到。

///////////////////////////
namespace svo {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
////////////////////////////////////////////////zbh
 fx=vk::getParam<int>("svo/cam_fx");
fy=vk::getParam<int>("svo/cam_fy");
cx=vk::getParam<int>("svo/cam_cx");
cy=vk::getParam<int>("svo/cam_cy");
Bf=vk::getParam<int>("svo/cam_Bf");
Depthsource=vk::getParam<std::string>("svo/Depth_source");
my_grad_size=vk::getParam<int>("svo/My_grad_size");
want_piont_num=vk::getParam<int>("svo/Want_piont_num");
constant_thred=vk::getParam<int>("svo/Constant_thred");
newkf_ratio=vk::getParam<float>("svo/newKF_ratio");
max_match=vk::getParam<int>("svo/max_fts");
depth_min_th=vk::getParam<int>("svo/depth_min_th");
depth_max_th=vk::getParam<int>("svo/depth_max_th");
align_only=vk::getParam<bool>("svo/align_only_");
display_Residuals=vk::getParam<bool>("svo/display_Residuals_");
gridsize_down_ratio=vk::getParam<float>("svo/gridsize_down_ratio_");
Align_every_level_by_ItsOwns_Point=vk::getParam<bool>("svo/Align_every_level_by_ItsOwns_Point_");
//以下参数不是数据成员
benmark_pyr_ratio=vk::getParam<float>("svo/pyr_ratio_");
benmark_change_pyr_ratio=vk::getParam<bool>("svo/change_pyr_ratio_");
Align_level_by_itsOwn_Point=vk::getParam<bool>("svo/Align_every_level_by_ItsOwns_Point_");
FastConer0_2_Grad4=vk::getParam<bool>("svo/FastConer0_2_Grad4_");
show_message=vk::getParam<bool>("svo/show_cput_Resident_mssage");

///////////////////////////////
 detector = new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels());
 this_time_keyframe=NULL;
DETAt=SE3(Matrix3d::Identity(), Vector3d::Zero());
  initialize();
}

void FrameHandlerMono::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  #ifndef USE_STEREO
  	printf("开始深度滤波线程！");
	depth_filter_->startThread();
	
  #endif
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}
//write by zhubaohua
Eigen::Vector3d FrameHandlerMono::getTtoWorld ()
{
Eigen::Vector3d tran=last_frame_->T_f_w_.inverse().translation();
return tran;
}
//write by zhubaohua
Sophus::SE3 FrameHandlerMono::getT_f_w_()
{
return last_frame_->T_f_w_;
}
//write by zhubaohua
Sophus::SE3  FrameHandlerMono::getTto_lastframe()
{
           Sophus::SE3 detaT=last_frame_->T_f_w_*new_frame_->T_f_w_.inverse();
	  return detaT;
}
void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  SVO_START_TIMER("pyramid_creation");
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  SVO_STOP_TIMER("pyramid_creation");

  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;

  // two-frame bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif

  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
//cout<<"svo:"<<new_frame_->T_f_w_.translation()[0]<<" "<<new_frame_->T_f_w_.translation()[1]<<" "<<new_frame_->T_f_w_.translation()[2]<<endl;
//cv::waitKey(1000);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }

  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
  ofstream f;
  f.open("/home/bunny/SLAM/output/small_sense1/svo_mono.txt", ios_base::out | ios_base::app);
  f<<new_frame_->T_f_w_.inverse().translation()[0]<<" "<< new_frame_->T_f_w_.inverse().translation()[1]<<" "<<new_frame_->T_f_w_.inverse().translation()[2]<<endl;
  f.close();
  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);
//printf("一个新的关键帧！！");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  if(img_align_n_tracked > 30)
  {
    SE3 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;
    FrameHandlerMono::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{

  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
  /*ofstream f;
  f.open("/home/bunny/SLAM/output/MH01/RGBD_sence_depth.txt", ios_base::out | ios_base::app);
  f<<new_frame_->id_<<":"<<scene_depth_mean<<endl;
  f.close();*/
  //getchar();
  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;
  }
  return true;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
cout<<" n "<<n<<endl;
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
cout<<"core_kfs_  ";
for(auto i:core_kfs_)
cout<<i->id_<<" "<<endl;
}





FrameHandlerBase::UpdateResult FrameHandlerMono::processRGBDFirstFrame(const cv::Mat& img_l, const cv::Mat& disp)
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  //if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    //return RESULT_NO_KEYFRAME;
  //特征提取 
  Features features;
  //detector->setExistingFeatures(new_frame_->fts_);
++keyf;
cout<<"创建第"<<keyf <<"个关键帧 "<<endl;

#ifdef use_grad
           //svo对齐思想提取梯度点  只提取level0-2
			detector->detectGrad_cell_level0_2( new_frame_.get(),new_frame_->img_pyr_, 
			  			 my_grad_size,want_piont_num, constant_thred,
						features,gridsize_down_ratio);//网格提取梯度点
cout<<"第 "<<keyf<<" 个关键帧提取的梯度点的个数"<<features.size()<<endl;

#else
detector->detect(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取特征点
#endif
  cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);


  float X,Y,Z;
  Vector3d xyz_w;
/////////////////////////////////////////////////////zhb    深度来源
if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }

int count=0;
//////////////////////////////////////////////////zhb
  for(auto it = features.begin();it!=features.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)//
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息
        new_frame_->addFeature(*it);
	point->addFrameRef(*it);
count++;
  } //cout<<"特征点数量:"<<new_frame_->fts_.size()<<endl;
cout<<"这些梯度点深度有效的点的个数:"<<count<<endl;

  //map_.point_candidates_.addCandidatePointToFrame(new_frame_);
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");

///////////////////////////上一个关键帧作为必须作为当前帧的overlaps
this_time_keyframe=new_frame_;
  new_frame_->last_key_frame=new_frame_;/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
  return RESULT_IS_KEYFRAME;

}

FrameHandlerBase::UpdateResult FrameHandlerMono::processRGBDFrame(const cv::Mat& img_l, const cv::Mat& disp)
{

cout<<"第 "<<++frame<< " 帧"<<endl;

  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;
////////////////////////////////////////////////zbh/////////////////////////////////////////////
 new_frame_->last_key_frame=this_time_keyframe; 
/////////////////////////////////////////////////////////////////////////////////////////////
  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, display_Residuals, false);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);
////////////////////////////////////////////////////
Eigen::Vector3d align_=new_frame_->T_f_w_.inverse().translation();
svo_inf<<"此帧经过图像对齐后的位姿 ："<<align_[0]<<" "<<align_[1]<<" "<<align_[2]<<endl;
  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
 //std::cout<<"经过重投影匹配到的点的数量："<<repr_n_new_references<<std::endl;

  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }


  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }

  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  //if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  //{
    //depth_filter_->addFrame(new_frame_);
  //  return RESULT_NO_KEYFRAME;
 // }
  //cout<<"repr_n_new_references:"<<repr_n_new_references<<endl;
#ifdef use_grad
  
if((!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)&&repr_n_new_references>newkf_ratio*max_match)
  {
    return RESULT_NO_KEYFRAME;
  }							
  
#else
 
if((!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)&&repr_n_new_references>newkf_ratio*max_match)
  {
    return RESULT_NO_KEYFRAME;
  }	
#endif   
                                                           
cout<<"创建第"<<++keyf <<"个关键帧 "<<endl;
svo_inf<<"创建第"<<keyf <<"个关键帧 "<<endl;
/********************************************************************************************/

  //特征提取
  Features features;

  
#ifdef use_grad
           //svo对齐思想提取梯度点  只提取level0-2
			detector->detectGrad_cell_level0_2( new_frame_.get(),new_frame_->img_pyr_, 
			  			 my_grad_size,want_piont_num, constant_thred,
						features,gridsize_down_ratio);//网格提取梯度点

cout<<"第 "<<keyf<<" 个关键帧提取的梯度点的个数"<<features.size()<<endl;

#else
detector->setExistingFeatures(new_frame_->fts_);
  detector->detect(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取特征点
cout<<"第 "<<keyf<<" 个关键帧提取的角点的个数"<<features.size()<<endl;
#endif

  cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);


  float X,Y,Z;
  Vector3d xyz_w;
/////////////////////////////zhb
 if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }
int cnt=0;
///////////////////////////////////////////zhb
  for(auto it = features.begin();it!=features.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  new_frame_->T_f_w_.inverse()*xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息
        new_frame_->addFeature(*it);
	point->addFrameRef(*it);
cnt++;
  } 
//cout<<"这些梯度点深度有效的点的个数:"<<cnt<<endl;
svo_inf<<"这些梯度点深度有效的点的个数:"<<cnt<<endl;
/***************************************************************************************************************/
  new_frame_->setKeyframe();
///////////////////////////上一个关键帧作为必须作为当前帧的overlaps
 this_time_keyframe=new_frame_;
  
//////////////////////////////////////////////////////////////////////////
  SVO_DEBUG_STREAM("New keyframe selected.");
  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
   size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

 

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    //depth_filter_->removeKeyframe(furthest_frame);// TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);
//printf("一个新的关键帧！！");
  return RESULT_IS_KEYFRAME;
}
//////////////////////////////////////////////////////////////////////////zhubaohua
FrameHandlerBase::UpdateResult FrameHandlerMono::processRGBDFrame_just_align(const cv::Mat& img_l, const cv::Mat& disp)
{
  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, display_Residuals, false);

  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);

Eigen::Vector3d tran=getTtoWorld();
ALIGN<<tran[0]<<" "<<tran[1]<<" "<<tran[2]<<endl;
//RGBD_LSD<<"第 "<<i<<" 帧相对世界位姿： "<<tran[0]<<" "<<tran[1]<<" "<<tran[2]<<endl;
//////////////////////////////////////////////////////
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);


  //特征提取 
  Features features;
  
#ifdef use_grad
if(Align_every_level_by_ItsOwns_Point)

detector->detectGrad_cell( new_frame_.get(),new_frame_->img_pyr_, 
  			 my_grad_size,want_piont_num, constant_thred,
			features,gridsize_down_ratio);//提取梯度点
else
detector->detectGrad_cell_level0_2( new_frame_.get(),new_frame_->img_pyr_, 
  			 my_grad_size,want_piont_num, constant_thred,
			features,gridsize_down_ratio);//提取梯度点
cout<<"第 "<<++frame<<" 个帧提取的梯度点的个数"<<features.size()<<endl;

#else
  detector->detect(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取特征点
cout<<"第 "<<++frame<<" 个帧提取的角点的个数"<<features.size()<<endl;
#endif
  cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);

  float X,Y,Z;
  Vector3d xyz_w;
  if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }
////////////////////////////////zbh
 //Features::iterator  mylist;
 // for(mylist=last_frame_->fts_.begin();mylist!=last_frame_->fts_.end();mylist++)
  //{
////delete (*mylist)->point;
//}


int cnt=0;
//////////////////////////////////////////////////////////
     new_frame_->fts_.clear();                 //清空new_frame_的特征点，使其只是提取的梯度点
  for(auto it = features.begin();it!=features.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  new_frame_->T_f_w_.inverse()*xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息 
        new_frame_->addFeature(*it);	
cnt++;
  } 

/***************************************************************************************************************/

  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");
  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);

  

  // init new depth-filters
  //depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    //depth_filter_->removeKeyframe(furthest_frame);// TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  //map_.addKeyframe(new_frame_);
//printf("一个新的关键帧！！");
  return RESULT_IS_KEYFRAME;
}

////////////////////////////////////////////////////////////ZHUBAOHAU
FrameHandlerBase::UpdateResult FrameHandlerMono::processRGBDFirstFrame_LSD(const cv::Mat& img_l, const cv::Mat& disp)
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());

  //特征提取 
  Features features;
  //detector->setExistingFeatures(new_frame_->fts_);
++keyf;
cout<<"创建第"<<keyf <<"个关键帧 "<<endl;
svo_inf<<"创建第"<<keyf <<"个关键帧 "<<endl;
#ifdef use_grad

//分层对齐提取梯度点
 detector->detectGrad_cell( new_frame_.get(),new_frame_->img_pyr_, 
	  			              my_grad_size,want_piont_num, constant_thred,
					        features,gridsize_down_ratio);//网格提取梯度点
cout<<"第 "<<keyf<<" 个关键帧提取的梯度点的个数"<<features.size()<<endl;
svo_inf<<"第 "<<keyf<<" 个关键帧提取的梯度点的个数"<<features.size()<<endl;
#else
detector->detect(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取特征点
#endif
  cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);


  float X,Y,Z;
  Vector3d xyz_w;
/////////////////////////////////////////////////////zhb    深度来源
if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }

int count=0;
//////////////////////////////////////////////////zhb
  for(auto it = features.begin();it!=features.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)//
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息
        new_frame_->addFeature(*it);
	point->addFrameRef(*it);
count++;
  } //cout<<"特征点数量:"<<new_frame_->fts_.size()<<endl;
cout<<"这些梯度点深度有效的点的个数:"<<count<<endl;
svo_inf<<"这些梯度点深度有效的点的个数:"<<count<<endl;
  //map_.point_candidates_.addCandidatePointToFrame(new_frame_);
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  
  return RESULT_IS_KEYFRAME;
}

///////////////////////////////////////////////////////////////ZHUBAOHUA
FrameHandlerBase::UpdateResult FrameHandlerMono::processRGBDFrame_LSD(const cv::Mat& img_l, const cv::Mat& disp)
{
svo_inf<<"第 "<<++frame<< " 帧"<<endl;
  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, display_Residuals, false);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
 //std::cout<<"经过重投影匹配到的点的数量："<<repr_n_new_references<<std::endl;
svo_inf<<"经过重投影匹配到的点的数量："<<repr_n_new_references<<std::endl;
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }


  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }

  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);

  //位姿优化后为当前帧提取梯度点，提取层level0-4
                                                           

/********************************************************************************************/

  //特征提取
  Features features;
  
#ifdef use_grad
          //分层对齐提取梯度点
		  detector->detectGrad_cell( new_frame_.get(),new_frame_->img_pyr_, 
	  			              my_grad_size,want_piont_num, constant_thred,
					        features,gridsize_down_ratio);//网格提取梯度点

cout<<"第 "<<frame<<" 个帧提取的梯度点的个数"<<features.size()<<endl;
svo_inf<<"第 "<<frame<<" 个帧提取的梯度点的个数"<<features.size()<<endl;
#else

  detector->detect(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取特征点
#endif
  cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);


  float X,Y,Z;
  Vector3d xyz_w;
/////////////////////////////zhb
 if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }
int cnt=0;
///////////////////////////////////////////zhb

new_frame_->fts_.clear();//把重投影匹配到的点清空，不让其作为特征点。

  for(auto it = features.begin();it!=features.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  new_frame_->T_f_w_.inverse()*xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息
        new_frame_->addFeature(*it);
	point->addFrameRef(*it);
cnt++;
  } 
//cout<<"这些梯度点深度有效的点的个数:"<<cnt<<endl;
//svo_inf<<"这些梯度点深度有效的点的个数:"<<cnt<<endl;
/***************************************************************************************************************/
//判断当前帧是否是关键帧，其实这个版本的图像对齐和重投影匹配已经彻底独立。关键帧的作用只是用来优化,决定当前帧的overlaps是哪些帧。
#ifdef use_grad
  
if((!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)&&repr_n_new_references>newkf_ratio*max_match)
  {
    return RESULT_NO_KEYFRAME;
  }							
  
#else
 
if((!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD))
  {
    return RESULT_NO_KEYFRAME;
  }	
#endif 
cout<<"创建第"<<++keyf <<"个关键帧 "<<endl;
svo_inf<<"创建第"<<keyf <<"个关键帧 "<<endl;

  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");
  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

 //cout<<"Config::maxNKfs(): "<<Config::maxNKfs()<<endl;

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);
//printf("一个新的关键帧！！");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processRGBDFirstFrame_level4change(const cv::Mat& img_l, const cv::Mat& disp)
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  
  //特征提取 
  Features features;
  detector->setExistingFeatures(new_frame_->fts_);
++keyf;
cout<<"创建第"<<keyf <<"个关键帧 "<<endl;

detector->detect(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取角点  来源：level0-2 
detector->detectGrad(new_frame_.get(),new_frame_->img_pyr_,4,features);//提取梯度点  来源：level4
cout<<"第 "<<keyf<<" 个关键帧提取的梯度点的个数"<<features.size()<<endl;

  cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);


  float X,Y,Z;
  Vector3d xyz_w;
/////////////////////////////////////////////////////zhb    深度来源
if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }

int count=0;
//////////////////////////////////////////////////zhb
  for(auto it = features.begin();it!=features.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)//
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息
        new_frame_->addFeature(*it);
	point->addFrameRef(*it);
count++;
  } //cout<<"特征点数量:"<<new_frame_->fts_.size()<<endl;
cout<<"这些梯度点深度有效的点的个数:"<<count<<endl;
svo_inf<<"这些梯度点深度有效的点的个数:"<<count<<endl;
  //map_.point_candidates_.addCandidatePointToFrame(new_frame_);
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  ///////////////////////////上一个关键帧作为必须作为当前帧的overlaps
this_time_keyframe=new_frame_;
  new_frame_->last_key_frame=new_frame_;/////////////////////////////////////////////////////////////
  return RESULT_IS_KEYFRAME;
}


FrameHandlerBase::UpdateResult FrameHandlerMono::processRGBDFrame_level4change(const cv::Mat& img_l, const cv::Mat& disp)
{
cout<<"第 "<<++frame<< " 帧"<<endl;
svo_inf<<"第 "<<frame<< " 帧"<<endl;
  // Set initial pose TODO use prior
new_frame_->T_f_w_ = last_frame_->T_f_w_;
////////////////////////////////////////////////zbh/////////////////////////////////////////////
 new_frame_->last_key_frame=this_time_keyframe; 
/////////////////////////////////////////////////////////////////////////////////////////////
  // sparse image align
  SVO_START_TIMER("sparse_img_align");

  //SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
     //                      30, SparseImgAlign::LevenbergMarquardt, display_Residuals, false);
SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                         30, SparseImgAlign::GaussNewton, display_Residuals, false);

  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);
////////////////////////////////////////////////////
Eigen::Vector3d align_=new_frame_->T_f_w_.inverse().translation();
svo_inf<<"此帧经过图像对齐后的位姿 ："<<align_[0]<<" "<<align_[1]<<" "<<align_[2]<<endl;
  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
 //std::cout<<"经过重投影匹配到的点的数量："<<repr_n_new_references<<std::endl;
svo_inf<<"经过重投影匹配到的点的数量："<<repr_n_new_references<<std::endl;
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }


  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }

  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  //if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  //{
    //depth_filter_->addFrame(new_frame_);
  //  return RESULT_NO_KEYFRAME;
 // }
  //cout<<"repr_n_new_references:"<<repr_n_new_references<<endl;
cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);
  float X,Y,Z;
  Vector3d xyz_w;
/////////////////////////////zhb
 if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }
//每帧提取level4的梯度点
  Features gradfeatures;
detector->detectGrad(new_frame_.get(),new_frame_->img_pyr_,4,gradfeatures);
//cout<<"当前帧提取的梯度点个数 : "<<gradfeatures.size()<<endl;
int count_grad_point=0;
  for(auto it = gradfeatures.begin();it!=gradfeatures.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  new_frame_->T_f_w_.inverse()*xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息
        new_frame_->addFeature(*it);
	point->addFrameRef(*it);
count_grad_point++;
  } 

//cout<<"当前帧梯度点经过深度筛选后有效的点数 ： "<<count_grad_point<<endl;
#ifdef use_grad
  
if((!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)&&repr_n_new_references>newkf_ratio*max_match)
  {
    return RESULT_NO_KEYFRAME;
  }							
  
#else
 
if((!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)&&repr_n_new_references>newkf_ratio*max_match)
  {
    return RESULT_NO_KEYFRAME;
  }	
#endif   
                                                           
cout<<"创建第"<<++keyf <<"个关键帧 "<<endl;
svo_inf<<"创建第"<<keyf <<"个关键帧 "<<endl;
/********************************************************************************************/

  //关键帧fast角点特征提取
  Features features;

//detector->setExistingFeatures(new_frame_->fts_);
  detector->detect(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取特征点
 //detector->detectGrad(new_frame_.get(),new_frame_->img_pyr_,Config::triangMinCornerScore(),features);//提取特征点
cout<<"第 "<<keyf<<" 个关键帧提取的FAST角点的个数"<<features.size()<<endl;


#ifdef deletee
  cv::Mat disparity(cam_->height(), cam_->width(),CV_32F);
  cv::Mat depth(cam_->height(), cam_->width(),CV_32F);
  float X,Y,Z;
  Vector3d xyz_w;
/////////////////////////////zhb
 if(Depthsource=="SGBM")
       {
          for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
		
		disparity.at<float>(i,j) = (float)disp.at<cv::Vec3b>(i,j)[1]/100 + disp.at<cv::Vec3b>(i,j)[0];
			  depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}
      }
 else if(Depthsource=="MoveSense")
       {
               for(int i=0;i<disp.rows;i++)
        	for(int j=0;j<disp.cols;j++)
     	         {
			      disparity.at<float>(i,j) = disp.at<uchar>(i,j)/4.0;
				depth.at<float>(i,j) = Bf/disparity.at<float>(i,j);
		}

      }
#endif
int cnt=0;

///////////////////////////////////////////zhb
  for(auto it = features.begin();it!=features.end();it++)
  {
        Z = depth.at<float>((*it)->px[1],(*it)->px[0]);
	if(Z<depth_min_th||Z>depth_max_th)
	    continue;
	//cout<<"depth:"<<Z<<endl;
        X = ((*it)->px[0]-cx)*Z/fx;
	Y = ((*it)->px[1]-cy)*Z/fy;
        //创建点
  	Vector3d xyz_f(X, Y, Z);
	xyz_w =  new_frame_->T_f_w_.inverse()*xyz_f;
  	Point* point = new Point(xyz_w, *it);//创建点
	(*it)->point = point;//为特征添加三维点信息
        new_frame_->addFeature(*it);
	point->addFrameRef(*it);
cnt++;
  } 
cout<<"这些角点深度有效的点的个数:"<<cnt<<endl;
svo_inf<<"这些梯度点深度有效的点的个数:"<<cnt<<endl;
/***************************************************************************************************************/
  new_frame_->setKeyframe();
///////////////////////////上一个关键帧作为必须作为当前帧的overlaps
 this_time_keyframe=new_frame_;
  
//////////////////////////////////////////////////////////////////////////
  SVO_DEBUG_STREAM("New keyframe selected.");
  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
cout<<"     LBA  now"<<endl;
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

 

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    //depth_filter_->removeKeyframe(furthest_frame);// TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);
//printf("一个新的关键帧！！");
  return RESULT_IS_KEYFRAME;
}



//////////////////////////////////////////////////////////////////////////

void FrameHandlerMono::addRGBDImage(const cv::Mat& img, const cv::Mat& disp, const double timestamp)
{

  if(!startFrameProcessingCommon(timestamp))
    return;
  //std::cout<<"running StereoImage!"<<std::endl;
  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();
  // create new frame
  SVO_START_TIMER("pyramid_creation");
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  SVO_STOP_TIMER("pyramid_creation");
//getchar();
  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    { 

       if(align_only)
	res =processRGBDFrame_just_align(img, disp);//只是图像对齐
       else if(Align_every_level_by_ItsOwns_Point)
         res = processRGBDFrame_LSD(img, disp);//图像对齐和重投影优化分离
       else if(FastConer0_2_Grad4)
	res = processRGBDFrame_level4change(img, disp);
       else
	res = processRGBDFrame(img, disp);//图像对齐和重投影优化  
//输出每帧相对于上一帧的位置变化detaTrans

	 DETAt=getTto_lastframe();
	Eigen::Vector3d deta_trans=DETAt.translation();

//输出每帧相对于上一帧的RPY变化detaR 
float R=0,P=0,Y=0;
 getRPY(new_frame_->T_f_w_,last_frame_->T_f_w_,R,P,Y);
	detaT<<deta_trans[0]<<" "<<deta_trans[1]<<" "<<deta_trans[2]<<" "<<R<<" "<<P<<" "<<Y<<endl;



    }
  else if(stage_ == STAGE_FIRST_FRAME)
	{

          if(Align_every_level_by_ItsOwns_Point)
             res = processRGBDFirstFrame_LSD(img, disp);//图像对齐和重投影优化分离
          else if(FastConer0_2_Grad4)
		res = processRGBDFirstFrame_level4change(img, disp);
          else
    	   res = processRGBDFirstFrame(img, disp);

  
	}
  else if(stage_ == STAGE_RELOCALIZING)
{
        getchar();
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

}  

  // set last frame
  last_frame_ = new_frame_;
//cout<<"last_frame_.use_count()  3: "<<last_frame_.use_count()<<endl;
  new_frame_.reset();
  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}


} // namespace svo
