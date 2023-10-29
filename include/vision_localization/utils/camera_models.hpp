#pragma once
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

// Pinhole [fx,fy,cx,cy]
// [u]   [fx  0  cx] [X/Z]
// [v] = [0  fy  cy] [Y/Z]
// [1]   [0   0   1] [1]

struct CameraInternalParameters {
    double fx;
    double fy;
    double cx;
    double cy;
    double ksi;
};

struct DistortedParamaters {
    double k1;
    double k2;
    double k3;
    double k4;
    double p1;
    double p2;
    double w;
};

// cam2world
void cam2World(const Eigen::Vector2d &cam, Eigen::Vector3d &world);

// world2cam
void world2cam(const Eigen::Vector3d &world, Eigen::Vector2d &cam);

// omnidirectional [ksi fx fy cx cy]
// 1、相机坐标系转到球面
//                       [Xs]
//  phai = [X, Y, Z] =>  [Ys] = phai/||phai||
//                       [Zs]

// 2、变换坐标系，新坐标系的原点位于Cp=(0,0,ksi)
//(Xs,Ys,Zs)Fm -->(Xs,Ys,Zs)Fp = (Xs,Ys,Zs+ksi)

// 3 并转换到归一化平面
// mu = (Xs/(Zs+ksi),Ys/(Zs+ksi),1) = h((Xs,Ys,Zs))

// 4 增加畸变影响
// md = mu + D(mu,V)

// 5 乘以内参矩阵K
// 无畸变
// [u]   [fx  0  cx]
// [v] = [0  fy  cy] * mu
// [1]   [0   0   1]
// 有畸变
// [u]   [fx  0  cx]
// [v] = [0  fy  cy] * md
// [1]   [0   0   1]

// cam2world
void omnicam2world(const Eigen::Vector2d &cam, Eigen::Vector3d &world);

// world2cam
void omniworld2cam(const Eigen::Vector3d &world, Eigen::Vector2d &cam);

// 畸变模型主要包括切向径向畸变(RadTan，radial-tangential
// distortion)、视野畸变(FOV,field of view)和等距畸变(Equidistant,EQUI)

//  Distortion models
//  Equidistant (EQUI) [k1,k2,k3,k4]
//  r = sqrt(xc*xc+yc*yc);
//  theta = atan2(r,|zc|) = atan2(r,1) = atan(r)
//  f = r' * tan(theta) r' = sqrt(u*u+v*v)
//  thetad = theta*(1+k1*theta*theta+k2*theta^4+k3*theta^6+k4*theta^8)
//  xd = thetad*xc/r;
//  yd = thetad*yc/r;

void equiModel(const Eigen::Vector3d &dis, Eigen::Vector3d &undis);

// Radtan [k1,k2,p1,p2]
// xdist = x(1+k1*r^2+k2*r^4) + 2p1xy+p2(r^2+2x^2)
// ydist = y(1+k1*r^2+k2*r^4) +p1(r^2+2x^2)+ 2p2xy

void radtanModel();

void radtanModel(const Eigen::Vector3d &dis, Eigen::Vector3d &undis);

// FOV [w]
// xdist = rd/r * x
// ydist = rd/r * y
// rd = 1/w*arctan(2*r*tan(w/2))
// r = sqrt((X/Z)^2 + (Y/Z)^2)= sqrt(x*x+y*y)

// MEI Camera = Omni + Radtan
// Pinhole Camera = Pinhole + Radtan
// atan Camera = Pinhole + FOV
// Davide Scaramuzza Camera:这个是ETHZ Davide
// Scaramuzza的工作，他将畸变和相机内参
// 放在一起了，为了克服针对鱼眼相机参数模型的知识缺乏，使用一个多项式来代替。

// 小于90度使用Pinhole，大于90度使用MEI模型。

// DSO：Pinhole + Equi / Radtan / FOV
// VINS：Pinhole / Omni + Radtan
// SVO：Pinhole / atan / Scaramuzza
// OpenCV：cv: pinhole + Radtan , cv::fisheye: pinhole + Equi , cv::omnidir:
// Omni + Radtan
