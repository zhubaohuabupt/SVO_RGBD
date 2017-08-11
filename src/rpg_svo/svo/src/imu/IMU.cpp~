#include "imu/IMU.h"
#include <iomanip>
namespace svo
{

IMU::IMU():imu_delta_t(0),earth_acc(0,0,-9.8),earth_omega(0,0,0)
{
 ///////////////////mh01   
#ifdef mh01
   q_imu2ground.w() = 0.541956;
    q_imu2ground.x() = -0.064676987;
    q_imu2ground.y() = 0.009810731;
    q_imu2ground.z() = -0.093543;
#endif
     q_imu2ground.w() = 0.592105;
    q_imu2ground.x() = 0.022393;
    q_imu2ground.y() = -0.805188;
    q_imu2ground.z() = 0.024127;

    v_old<<0,0,0;
    ba<<0,0,0;//0.0004,0.0004,0.0004;
    bg<<0,0,0;//0.005,0.005,0.005; 
    Matrix3d rotation;
    Vector3d translation(-0.021640145, -0.064676987, 0.0098107306);
    /*rotation<< 0.0148655429818, -0.999880929698, 0.00414029679422,
               0.999557249008, 0.0149672133247, 0.025715529948, 
               -0.0257744366974, 0.00375618835797, 0.999660727178;*/

    rotation<< 0.016350092, -0.99980514, 0.011061917,
               0.99971676, 0.016537997, 0.017113992,
                -0.017293599, 0.010778969, 0.99979235;
    T_imu_cam = new SE3(rotation, translation);
    T_imu_cam->translation() = translation;
    earth_acc_in_w = (q_imu2ground*T_imu_cam->unit_quaternion()).conjugate()._transformVector(earth_acc);//check
    Matrix<double,3,1> earth_acc_in_imu;
    earth_acc_in_imu = q_imu2ground.conjugate()._transformVector(earth_acc);//check
    //cout<<"earth_acc_in_w"<<earth_acc_in_w(0,0)<<","<<earth_acc_in_w(1,0)<<","<<earth_acc_in_w(2,0)<<endl;
    //cout<<"earth_acc_in_imu"<<earth_acc_in_imu(0,0)<<","<<earth_acc_in_imu(1,0)<<","<<earth_acc_in_imu(2,0)<<endl;
    pic_data.open("/media/baohua/study/927/salmdata/mav0_V202/pic_data/data.csv");
    imu_data.open("/media/baohua/study/927/salmdata/mav0_V202/imu_data/data.csv");
     imu_data_.open("/media/baohua/study/927/salmdata/mav0_V202/imu_data/data.csv");
    //imu_data_.open("/media/bunny/D2CC4F07CC4EE577/SLAM/IMU_DATA/MH01_data.csv");
    /*trans_to_Euler(q_imu2ground);
    cout<<"heading:"<<heading<<endl;
    cout<<"attitude:"<<attitude<<endl;
    cout<<"bank:"<<bank<<endl;*/
}

void IMU::load_imu_data()
{
    measurement.clear();
    measurement.reserve(7);
    char a[25],c[25];
    double b,t_temp;
    for(int i=0;i<7;i++)
    {
        if(i==6)
	{
	    imu_data.getline(a,25);
	    imu_data_.getline(c,25);
	}
	else
	{
	    imu_data.getline(a,25,',');
	    imu_data_.getline(c,25,',');
	}        
	if(i==0)
	{
	    t_temp = atof(c);
	}
	b = atof(a);
        measurement.push_back(b);
    }
    t_imu = measurement[0];
    imu_delta_t = t_temp - t_imu;
}

void IMU::calculate_pose(Frame* new_frame, Frame* last_frame)
{
    r_old = ((*T_imu_cam) * last_frame->T_f_w_).inverse().translation();
    q_old = ((*T_imu_cam) * last_frame->T_f_w_).inverse().unit_quaternion().conjugate();//
    get_pic_timestamp();
    while(t_imu+imu_delta_t<=pic_timestamp)
    {  
        if(pic_last_timestamp-t_imu<imu_delta_t&&pic_last_timestamp-t_imu>0)
	{
	    delta_t = (t_imu + imu_delta_t - pic_last_timestamp)*1e-9;
	    integration(last_measurement);
	    r_old = r_new;
	    v_old = v_new;
	    q_old = q_new;
	}
	else
	{
     		delta_t = imu_delta_t*1e-9;
		integration(measurement);
		r_old = r_new;
	        v_old = v_new;
	        q_old = q_new;	
	}
        last_imu_delta_t = imu_delta_t;
	load_imu_data();
    }
    if(t_imu+imu_delta_t>pic_timestamp)
    {
	delta_t = (pic_timestamp - t_imu)*1e-9;
	integration(measurement);
	r_old = r_new;
	v_old = v_new;
	q_old = q_new;
	last_measurement.clear();
	last_measurement = measurement;
    }
    pic_last_timestamp = pic_timestamp;
    new_frame->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
    new_frame->T_f_w_.setQuaternion(q_new.conjugate());
new_frame->T_f_w_.translation() = r_new;
//cout<<"newframe: "<<new_frame->T_f_w_.inverse().translation()[0]<<" "<<new_frame->T_f_w_.inverse().translation()[1]<<" "<<new_frame->T_f_w_.inverse().translation()[2]<<endl;
    new_frame->T_f_w_ = (new_frame->T_f_w_* (*T_imu_cam)).inverse();//.inverse()
/////////////////////////////////////////////////
   Eigen::Matrix3d R=new_frame->T_f_w_.inverse().rotation_matrix();
    Eigen::Vector3d T=last_frame->T_f_w_.inverse().translation();
     Sophus::SE3 T_w_f(R,T);
new_frame->T_f_w_=T_w_f.inverse();

/////////////////////////////////////////////////////////////////////
    //cout<<"v_new:"<<v_new(0,0)<<" "<<v_new(1,0)<<" "<<v_new(2,0)<<endl;
    /*T_test = SE3(Matrix3d::Identity(), Vector3d::Zero());
    T_test.setQuaternion(q_new.conjugate());
    T_test.translation() = r_new;
    T_test = (T_test * (*T_imu_cam)).inverse();*/
    /*Quaternion<double> qq;
    qq = new_frame->T_f_w_.unit_quaternion();
    trans_to_Euler(qq);
    ofstream file;
    file.open("/media/bunny/D2CC4F07CC4EE577/SLAM/TEST/SVO_IMU.txt",ios::app);
    file<<heading<<",  "<<attitude<<",  "<<bank<<endl;
    file.close();*/
    //getchar();
}

void IMU::integration(const vector<double>& imu_data)
{
    
    for(int i=0;i<3;i++)
    {    
        omega(i,0) = imu_data[i+1] - bg[i];
    	acc(i,0) = imu_data[i+4] - ba[i];
    }
    //cout<<"omega"<<omega(0,0)<<","<<omega(1,0)<<","<<omega(2,0)<<endl;
    //cout<<"acc"<<acc(0,0)<<","<<acc(1,0)<<","<<acc(2,0)<<endl;
    Quaternion<double> qe = rvec2quat(earth_omega*delta_t);
    q_new = q_old*qe;
    Quaternion<double> qb=rvec2quat(-omega*delta_t);//q_imu(new)_imu(old)
    q_new = qb*q_new;//q_imu(new)_w

    Eigen::Matrix<double,3,1> vel_inc1=(q_old.conjugate()._transformVector(acc*delta_t)+q_new.conjugate()._transformVector(acc*delta_t))*0.5;

    Eigen::Matrix<double,3,1> vel_inc2 = (earth_acc_in_w-2*earth_omega.cross(v_old))*delta_t;
    //cout<<"q_old:"<<q_old.w()<<","<<q_old.x()<<","<<q_old.y()<<","<<q_old.z()<<endl;
    //cout<<"acc:"<<(q_old.conjugate()._transformVector(acc))(0,0)<<","<<(q_old.conjugate()._transformVector(acc))(1,0)<<","<<(q_old.conjugate()._transformVector(acc))(2,0)<<endl;
    //cout<<"earth_acc:"<<earth_acc_in_w(0,0)<<","<<earth_acc_in_w(1,0)<<","<<earth_acc_in_w(2,0)<<endl;
    v_new = v_old+vel_inc1+vel_inc2;
    

    r_new = r_old+(v_new+v_old)*delta_t*0.5;
}

void IMU::sync()
{   
    
    t_imu = 0;
    char a[25];
    for(int i=0;i<7;i++)
    {
cout<<"getline"<<endl;
        if(i==6)
	    imu_data_.getline(a,25);
	
	else
	    imu_data_.getline(a,25,',');
    }

    get_pic_timestamp();

    load_imu_data();

    while(t_imu+imu_delta_t<=pic_timestamp)
	 load_imu_data();

    if(t_imu+imu_delta_t>pic_timestamp&&t_imu!=pic_timestamp)
	last_measurement = measurement;
    pic_last_timestamp = pic_timestamp;
   // cout<<fixed<<setprecision(25)<<"t_imu :"<<t_imu<<"pic_timestamp: "<<pic_timestamp<<endl;
//getchar();
}
void IMU::get_pic_timestamp()
{
    char a[25];
    double b;
    pic_data.getline(a,25,',');
    b = atof(a);
    pic_timestamp = b;
    pic_data.getline(a,25);//just aim to skip pic's name
}
void IMU::trans_to_Euler(Quaternion<double> q)
{
    double change = 3.1415926/180;
    if(q.x()*q.y()+q.z()*q.w()>0.499)
    {
	heading = 2*atan2(q.x(),q.w())/change;
	attitude = 3.1415926/2/change;
	bank = 0;
    }
    else if(q.x()*q.y()+q.z()*q.w()<-0.499)
    {
	heading = -2*atan2(q.x(),q.w())/change;
	attitude = -3.1415926/2/change;
	bank = 0;
    }
    else
    {    
    	heading = atan2(2*q.y()*q.w()-2*q.x()*q.z(),1-2*q.y()*q.y()-2*q.z()*q.z())/change;
    	attitude = asin(2*q.x()*q.y()+2*q.z()*q.w())/change;
    	bank = atan2(2*q.x()*q.w()-2*q.y()*q.z(),1-2*q.x()*q.x()-2*q.z()*q.z())/change;
    }
}
}
