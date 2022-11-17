#include <vector>

#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc_c.h>

#define PI 3.1415926
//#define RAD2DEG(x) (x)*180.0/PI
//#define DEG2RAD(x) (x)*PI/180.0
static const double WGS84_A = 6378137.0;      // major axis
static const double WGS84_E = 0.0818191908;   // first eccentricity


// ECEF转GPS
static cv::Point3d ECEF2LLA(cv::Point3d cur_ecef) {
    double x = cur_ecef.x;
    double y = cur_ecef.y;
    double z = cur_ecef.z;
    const double b = sqrt(WGS84_A * WGS84_A * (1 - WGS84_E * WGS84_E));
    const double ep = sqrt((WGS84_A * WGS84_A - b * b) / (b * b));
    const double p = hypot(x, y);
    const double th = atan2(WGS84_A * z, b * p);
    const double lon = atan2(y, x);
    const double lat = atan2((z + ep * ep * b * pow(sin(th), 3)), (p - WGS84_E * WGS84_E * WGS84_A * pow(cos(th), 3)));
    const double N = WGS84_A / sqrt(1 - WGS84_E * WGS84_E * sin(lat) * sin(lat));
    const double alt = p / cos(lat) - N;

    return cv::Point3d(RAD2DEG(lat), RAD2DEG(lon) ,alt);
}


// GPS转ECEF
static cv::Point3d LLA2ECEF(cv::Point3d cur_gps) {
    double lat = cur_gps.x;
    double lon = cur_gps.y;
    double alt = cur_gps.z;

    double WGS84_f = 1 / 298.257223565;
    double WGS84_E2 = WGS84_f * (2 - WGS84_f);
    double deg2rad = M_PI / 180.0;
    //double rad2deg = 180.0 / M_PI;
    lat *= deg2rad;
    lon *= deg2rad;
    double N = WGS84_A / (sqrt(1 - WGS84_E2 * sin(lat) * sin(lat)));
    double x = (N + alt) * cos(lat) * cos(lon);
    double y = (N + alt) * cos(lat) * sin(lon);
    double z = (N * (1 - WGS84_f) * (1 - WGS84_f) + alt) * sin(lat);

    return cv::Point3d(x, y, z);
}



// 地球半径
static const double EARTH_RADIUS = 6371000;

// 1、计算两个经纬度之间的距离(m)
static double GetDistance(cv::Point3d gps1, cv::Point3d gps2)
{
    double lat1 = gps1.x;
    double lon1 = gps1.y;

    double lat2 = gps2.x;
    double lon2 = gps2.y;

    double radLat1 = DEG2RAD(lat1);   // 角度转弧度
    double radLat2 = DEG2RAD(lat2);
    double a = radLat1 - radLat2;
    double b = DEG2RAD(lng1) - DEG2RAD(lng2);
    double s = 2 * asin(sqrt(pow(sin(a/2),2) +cos(radLat1)*cos(radLat2)*pow(sin(b/2),2)));
    s = s * EARTH_RADIUS;
    return s;
}

// 获取两个GPS坐标的纬度、经度、海拔差
static cv::Point3d GetDeltaGps(cv::Point3d gps1, cv::Point3d gps2)
{

    double lat1 = gps1.x;
    double lon1 = gps1.y;
    double alt1 = gps1.z;

    double lat2 = gps2.x;
    double lon2 = gps2.y;
    double alt2 = gps2.z;

    // 经度1秒 约等于 30.87*cos纬度（米）
    double delta_lon = abs(lon2-lon1)*3600*30.87*cos(lat1);

    // 纬度的1秒 = 30.8米
    double delta_lat = abs(lat2-lat1)*3600*30.8;

    double delta_alt = abs(alt2-alt1);

    return cv::Point3d(delta_lat, delta_lon, delta_alt);
}


// 格式化输出
template<typename ... Args>
static std::string str_format(const std::string &format, Args ... args)
{
	auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; 
	std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

	if (!buf)
		return std::string("");

	std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size_buf - 1); 
}


// 字符串分割
std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::vector<std::string> elems;
    auto lastPos = str.find_first_not_of(delim, 0);
    auto pos = str.find_first_of(delim, lastPos);
    while (pos != std::string::npos || lastPos != std::string::npos) {
        elems.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delim, pos);
        pos = str.find_first_of(delim, lastPos);
    }
    return elems;
}

// 数据读取
void LoadGPS(const string &strImuPath, vector<TXYZf> &vGPS, vector<long long> &vFrameTimes)
{
    ifstream fGps, fFrameTimes;

    fGps.open(strImuPath + "/GPS/gps.txt");     
    fFrameTimes.open(strImuPath + "/rgb/times.txt");

    vGPS.reserve(5000);
    vFrameTimes.reserve(5000);

    // 每一行数据使用'，'隔开
    while(!fGps.eof())
    {
        string s;
        getline(fGps, s);

        if(!s.empty())
        {
            vector<string> strs = stringSplit(s, ',');

            long long t = stoull(strs[0]);

            float lat, lon, alt;

            lat = stof(strs[2]);
            lon = stof(strs[3]);
            alt = stof(strs[4]);

            vGPS.push_back(TXYZf(t, lat, lon, alt));
        }
    }

    // 每一行数据使用空格隔开
    while(!fFrameTimes.eof())
    {
        string s;
        getline(fFrameTimes, s);

        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;

            long long t, t1, t2;
            ss >> t;
            ss >> t1;
            ss >> t2;

            vFrameTimes.push_back(t);
        }
    }

    fGps.close();
    fFrameTimes.close();
}