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

#ifndef SVO_FRAME_HANDLER_H_
#define SVO_FRAME_HANDLER_H_

#include <set>
#include <vikit/abstract_camera.h>
#include <svo/frame_handler_base.h>
#include <svo/reprojector.h>
#include <svo/feature_detection.h>
#include <svo/initialization.h>
#include"svo/SGM.h"
#include "imu/IMU.h"
#include"imu/eigen_utils.h"
namespace svo {

/// Monocular Visual Odometry Pipeline as described in the SVO paper.
class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  FrameHandlerMono(vk::AbstractCamera* cam);
  virtual ~FrameHandlerMono();

  
  feature_detection::FastDetector *detector;//writen by jio
 
  Eigen::Vector3d getTtoWorld();//zhubaohua
Sophus::SE3 getT_f_w_();//
  Sophus::SE3  getTto_lastframe();
////////////////////////////////zhubaohua
 float cx;
float cy;
float fx;
float fy;
float Bf;
std::string Depthsource;
int my_grad_size;//梯度提取网格尺寸
int want_piont_num;//期待获得梯度点的最大数量
int constant_thred;//弱纹理控制阈值
float newkf_ratio;//创建新的关键帧的参数1
int max_match;//创建新的关键帧的参数2
float depth_min_th;//筛选梯度点的条件 深度最小值
float depth_max_th;//筛选梯度点的条件 深度最大值
bool align_only;//是否要运行只图像对齐模式
bool display_Residuals;//是否要显示残差图
float gridsize_down_ratio;//提取梯度点时，相邻金字塔网格尺寸比值
bool Align_every_level_by_ItsOwns_Point;//是否要运行只图像对齐之每层特征点分别对齐模式
Sophus::SE3 DETAt;
///////////////////////////////zbh////////////////////

  /// Provide an image.
  void addImage(const cv::Mat& img, double timestamp);

  /// Provide RGBD image(writen by jio)
  void addRGBDImage(const cv::Mat& img, const cv::Mat& disp, const double timestamp);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  void setFirstFrame(const FramePtr& first_frame);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(
      const int keyframe_id,
      const SE3& T_kf_f,
      const cv::Mat& img,
      const double timestamp);

 MapPointCandidates* mappointcandidates;//writen by jio
protected:
 FramePtr this_time_keyframe;//////////////////////
//////////////////////////////

/////////////////////////////////
  vk::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.

  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();
  virtual UpdateResult processRGBDFirstFrame(const cv::Mat& img_l, const cv::Mat& disp);//writen by jio
  virtual UpdateResult processRGBDFrame(const cv::Mat& img_l, const cv::Mat& disp);//writen by jio
  virtual UpdateResult processRGBDFrame_just_align(const cv::Mat& img_l, const cv::Mat& disp);//writen by zbh
  virtual UpdateResult processRGBDFirstFrame_LSD(const cv::Mat& img_l, const cv::Mat& disp);//writen by zbh
  virtual UpdateResult processRGBDFrame_LSD(const cv::Mat& img_l, const cv::Mat& disp);//writen by zbh
   virtual UpdateResult processRGBDFirstFrame_level4change(const cv::Mat& img_l, const cv::Mat& disp);//writen by zbh
  virtual UpdateResult processRGBDFrame_level4change(const cv::Mat& img_l, const cv::Mat& disp);//writen by zbh
  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const SE3& T_cur_ref,
      FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(double scene_depth_mean);

  void setCoreKfs(size_t n_closest);
};

} // namespace svo

#endif // SVO_FRAME_HANDLER_H_
