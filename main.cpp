#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#pragma warning(disable : 4996)
using namespace std;
using namespace cv;

struct Point3D {
	double x, y, z;
};

int readData(char* path,vector<Point3D> &pointCloud) {
	ifstream file(path);
	if (!file.is_open()) {
		cout << "打开文件失败！" << endl;
		return 1;
	}

	string line;

	//跳过文件头
	while (getline(file, line)) {
		if (line == "end_header")
			break;
	}

	//读取点云数据
	while (getline(file, line)) {
		Point3D point;
		sscanf(line.c_str(), "%lf %lf %lf", &point.x, &point.y, &point.z);
		//你这点云数据怎么还有表面
		if (point.x < 0.001)
			break;
		pointCloud.push_back(point);
	}

	file.close();
}

//求点云的质心
Point3D getCenter(vector<Point3D> pointCloud) {
	Point3D center;
	center.x = 0;
	center.y = 0;
	center.z = 0;
	for (int i = 0; i < pointCloud.size(); i++) {
		center.x += pointCloud[i].x;
		center.y += pointCloud[i].y;
		center.z += pointCloud[i].z;
	}
	center.x /= pointCloud.size();
	center.y /= pointCloud.size();
	center.z /= pointCloud.size();
	return center;
}

//求各点相对于质心的位移向量
vector<Point3D> getDisplacement(vector<Point3D> pointCloud, Point3D center) {
	vector<Point3D> displacement;
	for (int i = 0; i < pointCloud.size(); i++) {
		Point3D point;
		point.x = pointCloud[i].x - center.x;
		point.y = pointCloud[i].y - center.y;
		point.z = pointCloud[i].z - center.z;
		displacement.push_back(point);
	}
	return displacement;
}


//vector做矩阵乘法
Mat cal_Mat_H(vector<Point3D> V1, vector<Point3D> V2) {
	Mat H = Mat::zeros(3, 3, CV_64FC1);
	Mat v1 = Mat::zeros(3, 1, CV_64FC1);
	Mat v2 = Mat::zeros(3, 1, CV_64FC1);
	for (int i = 0; i < V1.size(); i++)
	{
		//用点云的一个点对矩阵赋值
		v1.at<double>(0, 0) = V1[i].x;
		v1.at<double>(1, 0) = V1[i].y;
		v1.at<double>(2, 0) = V1[i].z;
		v2.at<double>(0, 0) = V2[i].x;
		v2.at<double>(1, 0) = V2[i].y;
		v2.at<double>(2, 0) = V2[i].z;
		H = H + v1 * v2.t();
	}
	return H;
}

Mat cal_Mat_V(Point3D center1, Point3D center2, Mat R) {
	Mat t = Mat::zeros(3, 1, CV_64FC1);
	Mat center1_Mat = Mat::zeros(3, 1, CV_64FC1);
	Mat center2_Mat = Mat::zeros(3, 1, CV_64FC1);
	center1_Mat.at<double>(0, 0) = center1.x;
	center1_Mat.at<double>(1, 0) = center1.y;
	center1_Mat.at<double>(2, 0) = center1.z;
	center2_Mat.at<double>(0, 0) = center2.x;
	center2_Mat.at<double>(1, 0) = center2.y;
	center2_Mat.at<double>(2, 0) = center2.z;
	t = center2_Mat - R * center1_Mat;
	return t;
}

//输出结果矩阵到文件
void Output(Mat R, Mat t) {
	ofstream file("./data/result.txt");
	if (!file.is_open()) {
		cout << "打开文件失败！" << endl;
		return;
	}
	file << "R = " << endl;
	file << R << endl;
	file << "t = " << endl;
	file << t << endl;
	file << endl;
	file << "适用于CloudCompare的变换矩阵为：" << endl;
	//规则输出
	for (int i = 0; i < R.rows; i++) {
		for (int j = 0; j < R.cols; j++) {
			file << setprecision(12) << R.at<double>(i, j) << " ";
		}
		file << setprecision(12) << t.at<double>(i, 0);
		file << endl;
	}
	file << "0.000000000000 0.000000000000 0.000000000000 1.000000000000" << endl;
	file.close();
}

double cal_RMSE(vector<Point3D> pointCloud1, vector<Point3D> pointCloud2, Mat R, Mat t) {
	//对CloudPoint1中每个点都进行一次R,t变换
	Mat point1 = Mat::zeros(3, 1, CV_64FC1);
	Mat point2 = Mat::zeros(3, 1, CV_64FC1);
	double tmp = 0, distance = 0;
	for (int i = 0; i < pointCloud1.size(); i++) {
		point1.at<double>(0, 0) = pointCloud1[i].x;
		point1.at<double>(1, 0) = pointCloud1[i].y;
		point1.at<double>(2, 0) = pointCloud1[i].z;
		point2.at<double>(0, 0) = pointCloud2[i].x;
		point2.at<double>(1, 0) = pointCloud2[i].y;
		point2.at<double>(2, 0) = pointCloud2[i].z;
		point1 = R * point1 + t;
		//计算距离残差(利用NORM_L2范数)
		double tmp = norm(point1 - point2);
		distance += tmp * tmp;
	}
	//计算RMSE
	double RMSE = sqrt(distance / pointCloud1.size());
}

int main() {
	vector<Point3D> pointCloud1;
	vector<Point3D> pointCloud2;
	//定义文件路径
	char path1[] = "./data/hand-low-tri.ply";
	//char path1[] = "./data/testLow.txt";
	char path2[] = "./data/trans-hand-low-tri.ply";
	//读取点云
	readData(path1, pointCloud1);
	readData(path2, pointCloud2);
	//计算重心
	Point3D center1 = getCenter(pointCloud1);
	Point3D center2 = getCenter(pointCloud2);
	//计算位移向量V
	vector<Point3D> displacement1 = getDisplacement(pointCloud1, center1);
	vector<Point3D> displacement2 = getDisplacement(pointCloud2, center2);
	//计算H矩阵
	Mat H = cal_Mat_H(displacement1, displacement2);
	//SVD分解H矩阵，求取旋转矩阵R
	Mat U, W, Vt;
	SVD::compute(H, W, U, Vt);
	Mat R = Vt.t() * U.t();
	//计算R的行列式，验证方法是否成功
	double det = determinant(R);
	//cout << "det(R) = " << det << endl;
	if (det < 0) {
		cout << "ICP方法失效，失效原因未知" << endl;
		return 0;
	}
	//计算平移向量t
	Mat t = cal_Mat_V(center1, center2, R);
	//输出刚体变换
	Output(R, t);
	//计算RMSE
	double RMSE = cal_RMSE(pointCloud1, pointCloud2, R, t);
	cout << "RMSE = " << RMSE << endl;
	return 0;
}