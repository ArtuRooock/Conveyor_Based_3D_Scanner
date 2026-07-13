#include <iostream>
#include <vector>
#include <Eigen/Eigen>

#include  "cnpy/cnpy.h"

using namespace Eigen;


static double baseline;
static double cx;
static double fx;
static double cy;
static double t_;
static std::vector<std::vector<double>> pts;



inline Matrix3d getRotMaty(double eta) {
    Matrix3d m;
    m << cos(eta),  0, sin(eta),
            0,         1, 0,
            -sin(eta), 0, cos(eta);
    return m;
}

inline Matrix3d getRotMatz(double eta) {
    Matrix3d m;
    m << cos(eta), -sin(eta), 0,
            sin(eta),  cos(eta), 0,
            0, 0, 1;
    return m;
}


inline Vector3d planeNorm(double eta, double theta) {
    Matrix3Xd m(3,2);
    m << 0, 0,
            -1, 0,
            0, 1;

    auto t =  getRotMaty(theta) * getRotMatz(eta) * m;
    return t.col(0).cross(t.col(1));
}

inline Vector3d rayVector(double x, double y, double f) {
    Vector3d v;
    v << x, y, f;

    return v;
}

static Vector3d getIntersectPoint(Vector3d ray, Vector3d planeN, double baseline) {
    auto t = planeN.transpose()*ray;
    if (t == 0) {
        return {-100, -100, -100};
    }
    return (baseline*planeN[0]/t) * ray;
}


double degConvert(double degree)
{
    double pi = 3.14159265359;
    return (degree * (pi / 180));
}


static double calcZ(double f, double x, double y) {
    const auto r = planeNorm(f, t_);
    const auto rv = rayVector(x, y, fx);
    const auto ip = getIntersectPoint(rv, r, baseline);
    const double at = atan(sqrt(x*x+y*y)/fx);

    return ip.norm() * cos(at);
}


static double opt(double fi) {
    double diff = 0;
    double fi_ = degConvert(fi);
    auto ptsn = pts[0];
    auto cz0 = calcZ(fi_, ptsn[3]-cx, ptsn[4]-cy);
    for (const auto & pt : pts) {
        const double e = (calcZ(fi_, pt[3]-cx, pt[4]-cy)-cz0) - (pt[2] - ptsn[2]);
        diff += sqrt(e*e);
    }

    return diff / pts.size();
}


int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Pass calib.camera npz, ba npz, and points.npz, res npz path" << std::endl;
        std::cerr << "opt_step" << std::endl;
        exit(1);
    }

    std::string camArr = argv[1];
    std::string baCalib = argv[2];
    std::string pointsArr = argv[3];
    std::string outputArch = argv[4];
    double optStep = atof(argv[5]);

    std::cout << "camArr = " << camArr << std::endl;
    std::cout << "BA calib arr" << baCalib << std::endl;
    std::cout << "pointsArr = " << pointsArr << std::endl;

    cnpy::NpyArray arr2 = cnpy::npz_load(camArr,"mtx");
    std::cout << "Shape: " << arr2.shape[0] << " " << arr2.shape[1] << std::endl;
    std::cout << "Word size: " << arr2.word_size << std::endl;
    if (arr2.word_size != 8) {
        std::cerr << "Expected np.float64" << std::endl;
        exit(2);
    }
    if (arr2.fortran_order) {
        std::cerr << "Not expected fortran order" << std::endl;
        exit(3);
    }

    auto *camMat = arr2.data<double>();
    cx = camMat[2];
    fx = camMat[0];
    cy = camMat[5];

    std::cout << "cx = " << cx << std::endl;
    std::cout << "cy = " << cy << std::endl;
    std::cout << "fx = " << fx << std::endl;

    cnpy::NpyArray baArr = cnpy::npz_load(baCalib,"ba");
    auto *baMat = baArr.data<double>();
    baseline = baMat[0];
    double t = baMat[1];
    t_ = degConvert(t);

    std::cout << "b = " << baseline << std::endl;
    std::cout << "th = " << t << " (" << t_ << ")" << std::endl;

    std::cout << "Loading points: ..." << std::endl;
    cnpy::npz_t pointsNpz = cnpy::npz_load(pointsArr);

    cnpy::NpyArray worldPoints = pointsNpz["psns"];
    cnpy::NpyArray camPoints = pointsNpz["cam_ptss"];

    if (camPoints.fortran_order || worldPoints.fortran_order) {
        std::cerr << "Not expected fortran order" << std::endl;
        exit(4);
    }

    std::cout << "WP Shape: " << worldPoints.shape[0] << " " << worldPoints.shape[1] << std::endl;
    std::cout << "CP Shape: " << camPoints.shape[0] << " " << camPoints.shape[1] << std::endl;

    if (worldPoints.shape[1] != 3 || camPoints.shape[1] != 2 || worldPoints.shape[0] != camPoints.shape[0]) {
        std::cerr << "Bad shape passed" << std::endl;
        exit(5);
    }

    auto *wPtr = worldPoints.data<double>();
    auto *cPtr = camPoints.data<double>();

    pts.resize(worldPoints.shape[0]);
    for (int i = 0; i < worldPoints.shape[0]; i++) {
        pts[i].resize(5);
        for (int j = 0; j < 3; j++) {
            pts[i][j] = wPtr[i*3+j];
        }
        for (int j = 0; j < 2; j++) {
            pts[i][j+3] = cPtr[i*2+j];
        }
        for (int j = 0; j < 5; j++) {
            std::cout << pts[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Optimization params: " << std::endl;
    std::cout << "PHI: from -89 to 89 step " << optStep << std::endl;

    int optSteps = (int)((89 * 2) / optStep);
    std::cout << "Will be performed: " << optSteps << " steps" << std::endl;
    std::vector<double> results(optSteps, 0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < results.size(); i++) {
        results[i] = opt(-89 + optStep*(double)i);
    }


    double mv = 9999999999;
    int index = 0;
    for (int i = 0; i < results.size(); i++) {
        if (mv > results[i]) {
            mv = results[i];
            index = i;
        }
    }

    const double fi = -89. + optStep*(double)index;
    std::cout << "Optimization done... Results: " << std::endl;
    std::cout << "Baseline: " << baseline << std::endl;
    std::cout << "Angle tetta: " << t << std::endl;
    std::cout << "Angle fi: " << fi << std::endl;
    std::cout << "Optimization score: " << mv << std::endl;

    std::vector<double> ba = {baseline, t, fi};
    cnpy::npz_save(outputArch,"baf",&ba[0],{3},"w");

    std::cout << "Results saved to: " << outputArch << std::endl;
    return 0;
}
