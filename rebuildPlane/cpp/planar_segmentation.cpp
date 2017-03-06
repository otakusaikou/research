#include <iostream>
#include <fstream>
#include <cstdlib>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

//
// Function for opening writable text file
//
std::fstream &openFile(std::fstream &fout, const std::string &fileName)
{
    fout.close();                    // Close in case the file was already open
    fout.clear();                    // Clear any existing errors
    fout.open(fileName.c_str(), std::fstream::out);     // Open the file with given file name
    return fout;
}

//
// Reference: https://goo.gl/UwVgM5
//
int main(int argc, char *argv[])

{
    // Read pcd file
    // for (int i = 0; i < argc; ++i) std::cout << argv[i] << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    if (argc < 2)
    {
        std::cerr << "You should specify one pcd file name" << std::endl;
        return -1;
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1)     // Load the file
    {
        PCL_ERROR("Couldn't read the input pcd file \n");
        return (-1);
    }

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    // Optional
    seg.setOptimizeCoefficients(true);

    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(std::atof(argv[3]));  // 0.01

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return -1;
    }

    // std::cout << coefficients->values[0] << " "
    //     << coefficients->values[1] << " "
    //     << coefficients->values[2] << " "
    //     << coefficients->values[3] << std::endl;

    std::fstream fout;
    openFile(fout, argv[2]);
    fout << "X Y Z" << std::endl;
    for (size_t i = 0; i < inliers->indices.size (); ++i)
        fout << cloud->points[inliers->indices[i]].x << " "
            << cloud->points[inliers->indices[i]].y << " "
            << cloud->points[inliers->indices[i]].z << std::endl;
    fout.close();

  return 0;
}
