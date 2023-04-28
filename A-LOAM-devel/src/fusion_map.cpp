#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <std_srvs/Trigger.h>
#include <list>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ceres/ceres.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <eigen3/Eigen/Dense>
#include <pcl/outofcore/outofcore.h>

#include <pcl/outofcore/outofcore_impl.h>
std::list<sensor_msgs::PointCloud2ConstPtr> CloudRegisteredBuf;
std::list<sensor_msgs::PointCloud2ConstPtr>::iterator Clouditer;
std::list<sensor_msgs::Image::ConstPtr> ImageBuf;
std::list<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::list<nav_msgs::Odometry::ConstPtr>::iterator Odoiter;
std::list<nav_msgs::Odometry::ConstPtr>::iterator Odoiter1;
std::mutex mBuf;
cv_bridge::CvImagePtr cv_ptr; // 声明一个CvImage指针的实例
Eigen::Quaterniond q_wodom_last(1, 0, 0, 0);
Eigen::Vector3d t_wodom_last(0, 0, 0);
Eigen::Quaterniond q_wodom_next(1, 0, 0, 0);
Eigen::Vector3d t_wodom_next(0, 0, 0);
Eigen::Quaterniond q_wodom_cur(1, 0, 0, 0);
Eigen::Vector3d t_wodom_cur(0, 0, 0);
int image_use_count = 0;
pcl::PointCloud<pcl::PointXYZRGB>  totalCloud;
pcl::PointCloud<pcl::PointXYZ> sumCloud;
int image_count=0;
std::vector<Eigen::Vector3d> pc_temp;
ros::Publisher pubcloud_test;
ros::Publisher pubcloud_all;
ros::Publisher pubimg_test;

void loadCalibrationData(Eigen::MatrixXd &P_rect_00,Eigen::MatrixXd &R_rect_00,Eigen::MatrixXd &RT) {
    //外参RT；
    RT(0,0)= 7.533745e-03;
    RT(0, 1) = -9.999714e-01;
    RT(0, 2) =-6.166020e-04;
    RT(0, 3) = -4.069766e-03;
    RT(1, 0) = 1.480249e-02;
    RT(1, 1) = 7.280733e-04;
    RT(1, 2) = -9.998902e-01;
    RT(1, 3) = -7.631618e-02;
    RT(2, 0) = 9.998621e-01;
    RT(2, 1) = 7.523790e-03;
    RT(2, 2) = 1.480755e-02;
    RT(2, 3) = -2.717806e-01;
    RT(3, 0) = 0.0;
    RT(3, 1) = 0.0;
    RT(3, 2) = 0.0;
    RT(3, 3) = 1.0;
    //内参；
    P_rect_00(0, 0) = 7.215377e+02;
    P_rect_00(0, 1) = 0.0;
    P_rect_00(0, 2) = 6.095593e+02;
    P_rect_00(0, 3) = 4.485728e+01;
    P_rect_00(1, 0) = 0;
    P_rect_00(1, 1) = 7.215377e+02;
    P_rect_00(1, 2) = 1.728540e+02;
    P_rect_00(1, 3) = 2.163791e-01;
    P_rect_00(2, 0) = 0.0;
    P_rect_00(2, 1) = 0.0;
    P_rect_00(2, 2) = 1.0;
    P_rect_00(2, 3) = 2.745884e-03;
    //相机之间的校准
    R_rect_00(0,0)=9.998817e-01;
    R_rect_00(0,1)=1.511453e-02;
    R_rect_00(0,2)=-2.841595e-03;
    R_rect_00(0,3)=0.0;
    R_rect_00(1,0)=-1.511724e-02;
    R_rect_00(1,1)=9.998853e-01;
    R_rect_00(1,2)=-9.338510e-04;
    R_rect_00(1,3)=0.0;
    R_rect_00(2,0)=2.827154e-03;
    R_rect_00(2,1)=9.766976e-04;
    R_rect_00(2,2)=9.999955e-01;
    R_rect_00(2,3)=0.0;
    R_rect_00(3,0)=0.0;
    R_rect_00(3,1)=0.0;
    R_rect_00(3,2)=0.0;
    R_rect_00(3,3)=1;

}

void projectLidarToCamera2(std::vector<Eigen::Vector3d> lidarPoints,pcl::PointCloud<pcl::PointXYZ> tempCloud,cv::Mat img) {


    Eigen::MatrixXd P_rect_00(3,4);
    Eigen::MatrixXd R_rect_00(4,4);
    Eigen::MatrixXd RT(4,4);
    loadCalibrationData(P_rect_00, R_rect_00,RT);


    // TODO: project lidar points
    image_count++;
    cv::Mat visImg = img.clone();
    cv::Mat overlay = visImg.clone();

    Eigen::Vector4d X;
    Eigen::Vector3d Y;
    pcl::PointCloud<pcl::PointXYZRGB> sumCloud1;

    int count =0;
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it) {

        X[0] = (*it)(0,0);
        X[1] = (*it)(1,0);
        X[2] = (*it)(2,0);
        X[3] = 1;
        double dis_sqr = X[0]*X[0]+X[1]*X[1]+X[2]*X[2];
        Y =  P_rect_00*R_rect_00 * RT * X;
        //Y =  P_rect_00 * RT * X;
        cv::Point pt;
        pt.x = Y[0] / Y[2];
        pt.y = Y[1] / Y[2];

        if(pt.x>=0 && pt.x<1241 && pt.y >= 0 && pt.y <376 && X[0]>0.01&& dis_sqr<20*20 ){      //kitti 1241*376
            //  if(pt.x>=0 && pt.x<1280 && pt.y >= 0 && pt.y <720 && X.at<double>(0, 0)>0.01 ){
            pcl::PointXYZRGB pt_color;
            pt_color.x = tempCloud.points[count].x;
            pt_color.y = tempCloud.points[count].y;
            pt_color.z = tempCloud.points[count].z;
            pt_color.r = img.at<cv::Vec3b>(pt.y,pt.x)[2];
            pt_color.g = img.at<cv::Vec3b>(pt.y,pt.x)[1];
            pt_color.b = img.at<cv::Vec3b>(pt.y,pt.x)[0];

            sumCloud1.points.push_back(pt_color);
        }
        count++;
    }
    std::cout<<"the pixel of the image in 320 55 is  "<<(int)(img.at<cv::Vec3b>(320,55)[2])<<std::endl;
    std::cout<<"the number of image is   "<<image_count<<std::endl;
    totalCloud+=sumCloud1;

    sensor_msgs::PointCloud2 test_cloud;
    pcl::toROSMsg(sumCloud1, test_cloud);

    test_cloud.header.frame_id = "/camera_init";
    pubcloud_test.publish(test_cloud);    // test_cloud 是添加了rgb信息的点云

//add tr
    sensor_msgs::PointCloud2 color_cloud;
    pcl::toROSMsg(totalCloud, color_cloud);

    color_cloud.header.frame_id = "/camera_init";
    pubcloud_all.publish(color_cloud);    // test_cloud 是添加了rgb信息的点云

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg(); //图像转msg
    pubimg_test.publish(msg);
}

void CloudRegisteredHandler(
        const sensor_msgs::PointCloud2ConstPtr &CloudRegistered) {
    mBuf.lock();
    CloudRegisteredBuf.push_back(CloudRegistered);
//   std::cout<<"the pcl size is "<<CloudRegisteredBuf.size()<<std::endl;
    mBuf.unlock();
}

void ImageHandler(
        const sensor_msgs::ImageConstPtr&Image) {
    image_use_count++;
    if(image_use_count >=2){
        image_use_count = 0;

        mBuf.lock();
        ImageBuf.push_back(Image);
        mBuf.unlock();
    }
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry) {
    mBuf.lock();
    odometryBuf.push_back(laserOdometry);
    mBuf.unlock();
}

void process() {

    while (1) {
        if((!ImageBuf.empty() )&& (!odometryBuf.empty()) && (!CloudRegisteredBuf.empty()) && (ImageBuf.size()>10))
        {
            std::cout <<" begin"<<" iamge "<<ImageBuf.size()<<" odom " <<odometryBuf.size()<<" clooud "<<CloudRegisteredBuf.size() <<" bool "<<(!ImageBuf.empty() && !odometryBuf.empty() && !CloudRegisteredBuf.empty() && ImageBuf.size()>5)<<std::endl;
            mBuf.lock();

            sensor_msgs::Image::ConstPtr imageData = ImageBuf.front();

            ImageBuf.pop_front();
            double image_timeCur = imageData->header.stamp.toSec();

            double odometry_timelast = 0;
            double odometry_timenext = 0;

            bool sign = false;
            for(Odoiter = odometryBuf.begin();
                (Odoiter != odometryBuf.end()) && ( (image_timeCur)>(*Odoiter)->header.stamp.toSec());Odoiter++){
                sign =true;
            }
            if(sign == true){
                Odoiter--;
            }

            odometry_timelast = (*Odoiter)->header.stamp.toSec();

            std::cout <<3.1<<"image"<<setprecision(10)<<image_timeCur<<std::endl;
            std::cout <<3.3<<"last"<<setprecision(10)<<odometry_timelast<<std::endl;
            std::cout<<3.2<<"lidarPoint"<<setprecision(10)<<CloudRegisteredBuf.front()->header.stamp.toSec()<<std::endl;
            q_wodom_last.x() = (*Odoiter)->pose.pose.orientation.x;
            q_wodom_last.y() = (*Odoiter)->pose.pose.orientation.y;
            q_wodom_last.z() = (*Odoiter)->pose.pose.orientation.z;
            q_wodom_last.w() = (*Odoiter)->pose.pose.orientation.w;
            t_wodom_last.x() = (*Odoiter)->pose.pose.position.x;
            t_wodom_last.y() = (*Odoiter)->pose.pose.position.y;
            t_wodom_last.z() = (*Odoiter)->pose.pose.position.z;
            Odoiter++;
            odometry_timenext = (*(Odoiter))->header.stamp.toSec();
            q_wodom_next.x() = (*(Odoiter))->pose.pose.orientation.x;
            q_wodom_next.y() = (*(Odoiter))->pose.pose.orientation.y;
            q_wodom_next.z() = (*(Odoiter))->pose.pose.orientation.z;
            q_wodom_next.w() = (*(Odoiter))->pose.pose.orientation.w;
            t_wodom_next.x() = (*(Odoiter))->pose.pose.position.x;
            t_wodom_next.y() = (*(Odoiter))->pose.pose.position.y;
            t_wodom_next.z() = (*(Odoiter))->pose.pose.position.z;



            while(odometryBuf.front()->header.stamp.toSec()<odometry_timelast){
                odometryBuf.pop_front();
            }

            mBuf.unlock();
            double s;
            s = (image_timeCur-odometry_timelast)/(odometry_timenext-odometry_timelast);
            Eigen::Quaterniond q_wodom_cur = q_wodom_last.slerp(s, q_wodom_next);
            Eigen::Matrix3d r_wodom_cur(q_wodom_cur);
            Eigen::Vector3d t_wodom_cur = t_wodom_last+s * (t_wodom_next-t_wodom_last);
            Eigen::Matrix3d r_wodom_cur_inv = r_wodom_cur.inverse();
            Eigen::Vector3d t_wodom_cur_inv = -r_wodom_cur_inv*t_wodom_cur;


            std::cout <<"s "<<s<<std::endl;
            mBuf.lock();
            while(image_timeCur-0.1>CloudRegisteredBuf.front()->header.stamp.toSec()){
                CloudRegisteredBuf.pop_front();
                std::cout<<"the time is clear"<<std::endl;
            }

            std::cout <<"timep "<<CloudRegisteredBuf.front()->header.stamp.toSec()<<std::endl;
            std::cout <<"empty "<<CloudRegisteredBuf.empty()<<std::endl;
            sumCloud.clear();

            for(Clouditer = CloudRegisteredBuf.begin();
                ((Clouditer != CloudRegisteredBuf.end()) && ( (image_timeCur+0.1)>=((*Clouditer)->header.stamp.toSec())));Clouditer++){
                pcl::PointCloud<pcl::PointXYZ> tempCloud;

                pcl::fromROSMsg(*(*Clouditer), tempCloud);
                sumCloud+=tempCloud;

            }
            std::cout <<"empty1 "<<CloudRegisteredBuf.empty()<<std::endl;
            mBuf.unlock();

            pc_temp.clear();
            for(int s =0;s<sumCloud.size();s++){
                Eigen::Vector3d pt(sumCloud.points[s].x,sumCloud.points[s].y,sumCloud.points[s].z);
                Eigen::Vector3d transform_pt = r_wodom_cur_inv*pt+t_wodom_cur_inv;
                pc_temp.push_back(transform_pt);
            }

            std::cout << "sum_size"<<sumCloud.size()<<std::endl;

            //当前新获取的图像保存在imageCur中
            try{
                cv_ptr =  cv_bridge::toCvCopy(imageData, sensor_msgs::image_encodings::BGR8); //将ROS消息中的图象信息提取，生成新cv类型的图象，复制给CvImage指针
            }
            catch(cv_bridge::Exception& e){  //异常处理
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            cv::Mat imageCur = cv_ptr->image;

            projectLidarToCamera2(pc_temp,sumCloud,imageCur);

        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fusion_map");
    ros::NodeHandle nh;
    ros::Subscriber subCloudRegistered = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 10000,CloudRegisteredHandler);
    std::cout <<1<<std::endl;
    ros::Subscriber subImage =nh.subscribe<sensor_msgs::Image>("sematic_image", 10,ImageHandler);


    ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 10000, laserOdometryHandler);


    pubcloud_test = nh.advertise<sensor_msgs::PointCloud2>("/sumcloud_test", 100); //sumcloud_test 是加rgb信息的点云
    pubcloud_all  = nh.advertise<sensor_msgs::PointCloud2>("/sumcloud_color", 100);
    pubimg_test = nh.advertise<sensor_msgs::Image>("/sumimg_test", 100);//sumimg是添加了激光点投影到图像的img


    std::thread mapping_process{process};

    ros::spin();

    return 0;
}