#include "spr.h"
#include "csv_utils.h"
#include "inih/INIReader.h"

#include <argparse/argparse.hpp>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/cloud_iterator.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>

namespace py = pybind11;

Eigen::MatrixXf read_pointcloud_from_pcd(std::string pcd_path)
{   
    pcl::PointCloud<pcl::PointXYZI>::Ptr Y(new pcl::PointCloud<pcl::PointXYZI>);
    if(pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *Y)== -1){
        py::print("Can't load pcd file " + pcd_path);
    }
    Eigen::MatrixXf result(Y->size(),3);
    for(int index=0; index < int(Y->size()); index++){
        result(index, 0) = Y->points[index].x;
        result(index, 1) = Y->points[index].y;
        result(index, 2) = Y->points[index].z;
    }
    return result;
}

class SPRPython{
public:
    SPRPython(const std::string ini_path){
        Y =  pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

        // load ini configuration:
        INIReader paramsIni(ini_path);
        if (paramsIni.ParseError() != 0) {
            py::print("Can't load parameters file" + ini_path);
            param_flag = false;
            return;
        }

        param_flag = true;

        spr_algorithm = new SPR(
        /*beta =*/paramsIni.GetFloat("SPR", "beta", 1.0),
        /*lambda0 =*/paramsIni.GetFloat("SPR", "lambda0", 1.0),
        /*tau0 =*/paramsIni.GetFloat("SPR", "tau0", 5.0),
        /*nMaxIterations =*/(std::size_t)paramsIni.GetInteger("SPR", "nMaxIterations", 100),
        /*tolerance =*/paramsIni.GetFloat("SPR", "tolerance", 1e-5),
        /*kNN =*/(std::size_t)paramsIni.GetInteger("SPR", "kNN", 20),
        /*wOutliers =*/paramsIni.GetFloat("SPR", "wOutliers", 0.1),
        /*annealingFactor =*/paramsIni.GetFloat("SPR", "annealingFactor", 0.97),
        /*normalization =*/paramsIni.GetBoolean("SPR", "normalization", true)
        );
    };

    SPRPython(const std::map<std::string, float> init_params)
    {
        Y =  pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        
        bool keys_flag = true;

        std::vector<std::string> ist_keys;
        std::vector<std::string> must_keys;
        must_keys.push_back("beta");
        must_keys.push_back("lambda0");
        must_keys.push_back("tau0");
        must_keys.push_back("nMaxIterations");
        must_keys.push_back("tolerance");
        must_keys.push_back("kNN");
        must_keys.push_back("wOutliers");
        must_keys.push_back("annealingFactor");
        must_keys.push_back("normalization");

        for(auto const& element : init_params){
            ist_keys.push_back(element.first);
        }

        for(auto const& must_key : must_keys){
            if(std::find(ist_keys.begin(), ist_keys.end(), must_key) != ist_keys.end()) {} 
            else {
                py::print(must_key);
                keys_flag = false;
            }
        }

        if(keys_flag){
            param_flag = true;

            spr_algorithm = new SPR(
            /*beta =*/init_params.at("beta"),
            /*lambda0 =*/init_params.at("lambda0"),
            /*tau0 =*/init_params.at("tau0"),
            /*nMaxIterations =*/(std::size_t)init_params.at("nMaxIterations"),
            /*tolerance =*/init_params.at("tolerance"),
            /*kNN =*/(std::size_t)init_params.at("kNN"),
            /*wOutliers =*/init_params.at("wOutliers"),
            /*annealingFactor =*/init_params.at("annealingFactor"),
            /*normalization =*/init_params.at("normalization")
            );
        }
        else{
            param_flag = false;
        }

    }
 
    SPRPython(){
        spr_algorithm = new SPR;
        Y =  pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        param_flag = true;
        py::print("Default parameters will be used to initialise the SPR");
    }

    bool load_joint_location_from_csv(std::string csv_path){
        if (!boost::filesystem::exists(csv_path) ){
            py::print("Can't load csv file " + csv_path);
            csv_flag = false;
            return false;
        }
        csv_flag = true;
        X = load_csv_to_eigen<Eigen::MatrixX3f>(csv_path);
        return true;
    };

    bool load_joint_location_from_np(const Eigen::MatrixX3f &data){
        csv_flag = true;
        X = data;
        return true;
    }

    Eigen::MatrixX3f get_joint_location() const{
        return X;
    }
    
    bool load_point_cloud_from_pcd(std::string pcd_path, std::string type="xyzi"){
        boost::algorithm::to_lower(type);
        if(type == "xyzi"){
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_with_intensity(new pcl::PointCloud<pcl::PointXYZI>);
            if(pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud_with_intensity)== -1){
                py::print("Can't load pcd file " + pcd_path);
                pcd_flag = false;
                return false;
            }
            else{
                Y->points.resize(cloud_with_intensity->size());
                for (size_t i = 0; i < cloud_with_intensity->points.size(); i++) {
                    Y->points[i].x = cloud_with_intensity->points[i].x;
                    Y->points[i].y = cloud_with_intensity->points[i].y;
                    Y->points[i].z = cloud_with_intensity->points[i].z;
                }
                py::print("Convert XYZI point into XYZ point");
                pcd_flag = true;
                return true;
            }
        }else if(type == "xyz"){
            if(!pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *Y)){
                py::print("Can't load pcd file " + pcd_path);
                pcd_flag = false;
                return false;
            }else{
                pcd_flag = true;
                return true;
            }
        }else{
            py::print("Unsupported data type of PCD file, please make sure that your PCD file in XYZ or XYZI format");
            pcd_flag = false;
            return false;
        }
    };

    bool load_point_cloud_from_np(const Eigen::MatrixXf &data){
        Y->points.resize(data.rows());
        for (size_t i = 0; i < Y->points.size(); i++) {
            Y->points[i].x = data(i,0);
            Y->points[i].y = data(i,1);
            Y->points[i].z = data(i,2);
        }
        pcd_flag = true;
        return true;
    }

    void set_verbose(bool flag){
        if (!param_flag){
            py::print("The initial parameters for SPR algorithm have not been correctly loaded, please load them first");
            return;
        }
        if (!pcd_flag){
            py::print("The PCD file has not been correctly loaded, please load it");
            return;
        }
        if (!csv_flag){
            py::print("The initial joint location has not been correctly loaded, please load it");
            return;
        }
        spr_algorithm->setVerbose(flag);
    };
    
    bool compute_registered(){
        if (!param_flag){
            py::print("The initial parameters for SPR algorithm have not been correctly loaded, please load them first");
            return false;
        }
        if (!pcd_flag){
            py::print("The PCD file has not been correctly loaded, please load it");
            return false;
        }
        if (!csv_flag){
            py::print("The initial joint location has not been correctly loaded, please load it");
            return false;
        }
        X_registered = spr_algorithm->computeEM(X, Y);
        return true;
    };

    void save_registered_to_csv(std::string save_path){
        if(param_flag && csv_flag && pcd_flag){
            write_eigen_to_csv(save_path, X_registered);
        }
        else
        {
            py::print("The registered joint location has not been computed");
        }  
    }

    Eigen::MatrixX3f get_registered_joint_location() const{
        return X_registered;
    }

public:
    SPR* spr_algorithm;
    Eigen::MatrixX3f X;
    pcl::PointCloud<pcl::PointXYZ>::Ptr Y;
    Eigen::MatrixX3f X_registered;
    bool param_flag;
    bool csv_flag;
    bool pcd_flag;
};
