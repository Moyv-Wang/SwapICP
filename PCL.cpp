#include <iostream>
#include <ctime>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
using namespace cv;
using namespace std;

void readData(char* path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::PLYReader reader;
    reader.read<pcl::PointXYZ>(path, *cloud);
}

pcl::PointXYZ getCenter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::PointXYZ center;
	center.x = 0;
	center.y = 0;
	center.z = 0;
    for (int i = 0; i < cloud->size(); i++) {
		center.x += cloud->points[i].x;
		center.y += cloud->points[i].y;
        center.z += cloud->points[i].z;
	}
	center.x /= cloud->size();
	center.y /= cloud->size();
	center.z /= cloud->size();
    return center;
}

//�������������ĵ�λ������
pcl::PointCloud<pcl::PointXYZ>::Ptr getDisplacement(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ center) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr displacement(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ point;
    for (int i = 0; i < cloud->size(); i++) {
		point.x = cloud->points[i].x - center.x;
		point.y = cloud->points[i].y - center.y;
		point.z = cloud->points[i].z - center.z;
		displacement->push_back(point);
	}
    return displacement;
}


void cal_Centered_XYZ(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointXYZ center) {
    for (int i = 0; i < cloud->size(); i++) {
        cloud->points[i].x -= center.x;
        cloud->points[i].y -= center.y;
        cloud->points[i].z -= center.z;
    }
}

double cal_Scale(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2) {
    //�㷨ԭ�����������Ļ������������ȣ�����ֵ�����õ��߶�
    double scale = 0;
    double factor1 = 0, factor2 = 0;
    for (int i = 0; i < cloud1->size(); i++) {
        factor1 += pow(cloud1->points[i].x, 2) + pow(cloud1->points[i].y, 2) + pow(cloud1->points[i].z, 2);
        factor2 += pow(cloud2->points[i].x, 2) + pow(cloud2->points[i].y, 2) + pow(cloud2->points[i].z, 2);
    }
    scale = sqrt(factor1 / factor2);
    return scale;
}

void ScalingPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double scale) {
    //��ÿ�����������гߴ�任
    for (int i = 0; i < cloud->size(); i++) {
		cloud->points[i].x *= scale;
		cloud->points[i].y *= scale;
		cloud->points[i].z *= scale;
	}
}

Mat cal_Mat_H(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2) {
    Mat H = Mat::zeros(3, 3, CV_64FC1);
    Mat v1 = Mat::zeros(3, 1, CV_64FC1);
    Mat v2 = Mat::zeros(3, 1, CV_64FC1);
    for (int i = 0; i < cloud1->size(); i++) {
		v1.at<double>(0, 0) = cloud1->points[i].x;
		v1.at<double>(1, 0) = cloud1->points[i].y;
		v1.at<double>(2, 0) = cloud1->points[i].z;
		v2.at<double>(0, 0) = cloud2->points[i].x;
		v2.at<double>(1, 0) = cloud2->points[i].y;
		v2.at<double>(2, 0) = cloud2->points[i].z;
        H += v2 * v1.t();
	}
    return H;
}

Mat cal_Mat_V(pcl::PointXYZ center1, pcl::PointXYZ center2, Mat R) {
    Mat t = Mat::zeros(3, 1, CV_64FC1);
    Mat center1_Mat = Mat::zeros(3, 1, CV_64FC1);
    Mat center2_Mat = Mat::zeros(3, 1, CV_64FC1);
    center1_Mat.at<double>(0, 0) = center1.x;
    center1_Mat.at<double>(1, 0) = center1.y;
    center1_Mat.at<double>(2, 0) = center1.z;
    center2_Mat.at<double>(0, 0) = center2.x;
    center2_Mat.at<double>(1, 0) = center2.y;
    center2_Mat.at<double>(2, 0) = center2.z;
    t = center1_Mat - R * center2_Mat;
    return t;
}

double cal_RMSE(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2) {
    double tmp = 0, distance = 0;
    double RMSE = 0;
    int range = cloud1->size() > cloud2->size() ? cloud2->size() : cloud1->size();
    for (int i = 0; i < range; i++) {
        tmp = pow(cloud1->points[i].x - cloud2->points[i].x, 2) + pow(cloud1->points[i].y - cloud2->points[i].y, 2) + pow(cloud1->points[i].z - cloud2->points[i].z, 2);
		distance += sqrt(tmp);
	}
    RMSE = distance / cloud1->size();
	return RMSE;
}

//�����������ļ�
void Output(Mat R, Mat t, double RMSE, double Scale) {
    ofstream file("./data/New_Result.txt");
    if (!file.is_open()) {
        cout << "���ļ�ʧ�ܣ�" << endl;
        return;
    }
    file << "��������� RMSE = " << RMSE << endl;
    file << endl;
    file << "�߶ȱ任 Scale = " << Scale << endl;
    file << endl;
    file << "��ת���� R = " << endl;
    file << R << endl;
    file << endl;
    file << "ƽ������ t = " << endl;
    file << t << endl;
    file << endl;
    file << "������CloudCompare�ı任����Ϊ(Ҫ�Ƚ��г߶ȱ任)��" << endl;
    //�������
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

pcl::PointCloud<pcl::PointXYZ>::Ptr DownSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledCloud(new pcl::PointCloud<pcl::PointXYZ>);
    // �������ظ��²�������
    pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
    voxelGrid.setInputCloud(cloud);
    // �������ظ�Ĵ�С�����ر߳���
    voxelGrid.setLeafSize(0.01f, 0.01f, 0.01f); // ��������Ϊ 1cm x 1cm x 1cm
    // ִ�н�����
    voxelGrid.filter(*downsampledCloud);
    // ��ӡ������ǰ��ĵ��ƴ�С
    std::cout << "ԭʼ���ƴ�С: " << cloud->size() << std::endl;
    std::cout << "����������ƴ�С: " << downsampledCloud->size() << std::endl;
    return downsampledCloud;
}

int main()
{
    //��¼����ʼʱ��
    clock_t start = clock();
    clock_t read_start = clock();
    char path1[] = "./data/hand-low-tri.ply";
	char path2[] = "./data/scaled-hand-low-tri.ply";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    cout << "reading point cloud data..." << endl;
    readData(path1, cloud1);
    readData(path2, cloud2);
    cout << "reading finished" << ", time used��" << 1000.0 * (clock() - read_start) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "calculating Scale param..." << endl;
    clock_t scale_start = clock();
    pcl::PointXYZ center1 = getCenter(cloud1);
    pcl::PointXYZ center2 = getCenter(cloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr displacement1 = getDisplacement(cloud1, center1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr displacement2 = getDisplacement(cloud2, center2);
    //ȷ���������֮��ĳ߶Ȳ������ָ���������Ƶĳߴ�
    double scale = cal_Scale(displacement1, displacement2);
    cout << "Scale calculating finished" << ", time used��" << 1000.0 * (clock() - scale_start) / (double)CLOCKS_PER_SEC << "ms" << endl;
    ScalingPointCloud(cloud2, scale);
    cout << "Applying ICP algorithm..." << endl;
    clock_t ICP_start = clock();
    //��cloud�����²���
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledCloud1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledCloud2(new pcl::PointCloud<pcl::PointXYZ>);
    cloud1 = DownSampling(cloud1);
    cloud2 = DownSampling(cloud2);
    //��cloud1����kd-tree
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree = new pcl::KdTreeFLANN <pcl::PointXYZ>();
    kd_tree.setInputCloud(cloud1);
    double RMSE = 1;
    int flag = 0;
    //ȫ�ֱ任
    Mat global_R = Mat::eye(3, 3, CV_64FC1);
    Mat global_t = Mat::zeros(3, 1, CV_64FC1);
    do {
        //��cloud2��ÿ���㶼��cloud1����������ڵ㣬�����浽�µĵ�����
        pcl::PointCloud<pcl::PointXYZ>::Ptr NewPointCloud1(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr NewPointCloud2(new pcl::PointCloud<pcl::PointXYZ>);
        vector<int> index(1);
        vector<float> squredDistance(1);
        for (int i = 0; i < cloud2->size(); i++) {
            kd_tree.nearestKSearch(cloud2->points[i], 1, index, squredDistance);
            //�趨squaredDistance��ֵ
            if (squredDistance[0] > 1.3) {
				continue;
			}
            NewPointCloud1->push_back(cloud1->points[index[0]]);
            NewPointCloud2->push_back(cloud2->points[i]);
        }
        pcl::PointXYZ NewCenter1 = getCenter(NewPointCloud1);
        pcl::PointXYZ NewCenter2 = getCenter(NewPointCloud2);
        pcl::PointCloud<pcl::PointXYZ>::Ptr NewDis1 = getDisplacement(NewPointCloud1, NewCenter1);
        pcl::PointCloud<pcl::PointXYZ>::Ptr NewDis2 = getDisplacement(NewPointCloud2, NewCenter2);
        //����H����
        Mat H = cal_Mat_H(NewDis1, NewDis2);
        //SVD�ֽ�H������ȡ��ת����R
        Mat U, W, Vt;
        SVD::compute(H, W, U, Vt);
        Mat R = Vt.t() * U.t();
        //����R������ʽ����֤�����Ƿ�ɹ�
        double det = determinant(R);
        //cout << "det(R) = " << det << endl;
        if (det < 0) {
            cout << "ICP����ʧЧ���˳�����" << endl;
            return 0;
        }
        global_R = R * global_R;
        //����ƽ������t
        Mat t = cal_Mat_V(NewCenter1, NewCenter2, R);
        if(flag == 0)
            global_t = t;
        else
            global_t = R * global_t + t;
        //��cloud2�е�ÿ���������ת��ƽ��
        Mat v = Mat::zeros(3, 1, CV_64FC1);
        //cout << R << endl << t << endl;
        pcl::PointXYZ p;
        for (int i = 0; i < cloud2->size(); i++) {
            v.at<double>(0, 0) = cloud2->points[i].x;
            v.at<double>(1, 0) = cloud2->points[i].y;
            v.at<double>(2, 0) = cloud2->points[i].z;
            //cout << v << endl;
            v = R * v + t;
            //��p��ֵ
            p.x = v.at<double>(0, 0);
            p.y = v.at<double>(1, 0);
            p.z = v.at<double>(2, 0);
            //��p�滻��ǰ������ĵ�
            cloud2->points[i] = p;
            //NewPointCloud2->push_back(pcl::PointXYZ(v.at<double>(0, 0), v.at<double>(1, 0), v.at<double>(2, 0)));
        }
        //����RMSE
        RMSE = cal_RMSE(cloud1, cloud2);
        flag++;
        //���cloud2��./data/cloud2.ply
        //pcl::io::savePLYFileASCII("./data/cloud2.ply", *cloud2);
    } while (RMSE > 0.01);
    cout << "ICP finished, ����������"<< flag << ", RMSE��"<< RMSE << ", time used��" << 1000.0 * (clock() - ICP_start) / (double)CLOCKS_PER_SEC << "ms" << endl;
	cout << "total time used��" << 1000.0 * (clock() - start) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "���������浽./data/result.txt" << endl;
	//�������任
	Output(global_R, global_t, RMSE, scale);
    //cout << global_R << endl << global_t << endl;
    return 0;
}