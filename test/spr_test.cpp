#include "spr.h"
#include "csv_utils.h"

#include "inih/INIReader.h"
#include <argparse/argparse.hpp>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>


template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr load_pcd_file(const std::string &path) {
  typename pcl::PointCloud<PointT>::Ptr cloud(new typename pcl::PointCloud<PointT>);

  if (pcl::io::loadPCDFile<PointT>(path, *cloud) == -1) //* load the file
  {
    PCL_ERROR("Couldn't read file test_pcd.pcd \n");
  }
  return cloud;
}

template <typename PointT>
pcl::PointCloud<pcl::PointXYZ>::Ptr
convertPointCloudToXYZ(typename pcl::PointCloud<PointT>::Ptr pointCloudToConvert) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
  cloudXYZ->points.resize(pointCloudToConvert->size());

  for (size_t i = 0; i < pointCloudToConvert->points.size(); i++) {
    cloudXYZ->points[i].x = pointCloudToConvert->points[i].x;
    cloudXYZ->points[i].y = pointCloudToConvert->points[i].y;
    cloudXYZ->points[i].z = pointCloudToConvert->points[i].z;
  }
  return cloudXYZ;
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("spr_test");
  program.add_argument("X").help(
      "Path to csv containing intitial joint locations, each line the x,y,z of one joint");
  program.add_argument("Y").help("Path to pcd file containing measured point cloud");
  program.add_argument("--output", "-o")
      .default_value(std::string("Xregistered.csv"))
      .help("Name of output csv containing locations (default: Xregistered.csv)");
  program.add_argument("-v", "--verbose")
      .help("increase output verbosity")
      .default_value(false)
      .implicit_value(true);
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }

  Eigen::MatrixX3f X = load_csv_to_eigen<Eigen::MatrixX3f>(program.get<std::string>("X"));
  auto pc_with_intensities = load_pcd_file<pcl::PointXYZI>(program.get<std::string>("Y"));
  if (pc_with_intensities->size() == 0) {
    std::cout << "Unable to load Point Cloud at " << program.get<std::string>("Y") << ". Abort."
              << std::endl;
    return -1;
  }

  auto Y_cloud = convertPointCloudToXYZ<pcl::PointXYZI>(pc_with_intensities);

  // load ini configuration:
  INIReader paramsIni("params.ini");
  if (paramsIni.ParseError() != 0) {
    std::cout << "Can't load parameters file (SPRparams.ini)\n";
    return -2;
  }

  SPR spr_algorithm(
      /*beta =*/paramsIni.GetFloat("SPR", "beta", 1.0),
      /*lambda0 =*/paramsIni.GetFloat("SPR", "lambda0", 1.0),
      /*tau0 =*/paramsIni.GetFloat("SPR", "tau0", 5.0),
      /*nMaxIterations =*/(std::size_t)paramsIni.GetInteger("SPR", "nMaxIterations", 100),
      /*tolerance =*/paramsIni.GetFloat("SPR", "tolerance", 1e-5),
      /*kNN =*/(std::size_t)paramsIni.GetInteger("SPR", "kNN", 20),
      /*wOutliers =*/paramsIni.GetFloat("SPR", "wOutliers", 0.1),
      /*annealingFactor =*/paramsIni.GetFloat("SPR", "annealingFactor", 0.97),
      /*normalization =*/paramsIni.GetBoolean("SPR", "normalization", true));

  if (program.get<bool>("--verbose")) {
    spr_algorithm.setVerbose(true);
  }

  Eigen::MatrixX3f X_registered = spr_algorithm.computeEM(X, Y_cloud);
  write_eigen_to_csv(program.get<std::string>("--output"), X_registered);

  return 0;
}