#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <Eigen/Eigen>
 
#include  "cnpy/cnpy.h"
 
using namespace Eigen;
 
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
 
 
static std::vector<std::vector<double>> pts;
static double cx;
static double fx;
 
  
static double calcZ(double th, double b, double x) {
    const auto r = planeNorm(degConvert(0), th);
    const auto rv = rayVector(x, 0, fx);
    const auto ip = getIntersectPoint(rv, r, b);
    const double a = atan(x/fx);
 
    return ip.norm() * cos(a);
}
 

static double opt(double th_deg, double b) {
    double diff = 0;
    double t_ = degConvert(th_deg);
    for (const auto & pt : pts) {
        const double e = calcZ(t_, b, pt[3]-cx) - pt[2];
        diff += std::fabs(e);
    }
 
    return diff / pts.size();
}
 
 
static inline double clampVal(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}
 
 
struct Result {
    double theta;
    double baseline;
    double error;
};
 

static Result patternSearch(double th0, double b0,
                             double thMin, double thMax,
                             double bMin, double bMax,
                             double thStep0, double bStep0,
                             double thTol, double bTol) {
    double th = clampVal(th0, thMin, thMax);
    double b  = clampVal(b0, bMin, bMax);
    double thStep = thStep0;
    double bStep  = bStep0;
    double best = opt(th, b);
 
    while (thStep > thTol || bStep > bTol) {
        double candTh[4] = {
            clampVal(th + thStep, thMin, thMax),
            clampVal(th - thStep, thMin, thMax),
            th,
            th
        };
        double candB[4] = {
            b,
            b,
            clampVal(b + bStep, bMin, bMax),
            clampVal(b - bStep, bMin, bMax)
        };
 
        int bestIdx = -1;
        double bestErr = best;
        for (int i = 0; i < 4; i++) {
            const double e = opt(candTh[i], candB[i]);
            if (e < bestErr) {
                bestErr = e;
                bestIdx = i;
            }
        }
 
        if (bestIdx >= 0) {
            th = candTh[bestIdx];
            b  = candB[bestIdx];
            best = bestErr;
        } else {
            if (thStep > thTol) thStep /= 2.0;
            if (bStep  > bTol)  bStep  /= 2.0;
        }
    }
 
    return {th, b, best};
}
 
 
int main(int argc, char **argv) {
 
    if (argc != 9) {
        std::cerr << "Pass calib.camera npz and points.npz, res npz path" << std::endl;
        std::cerr << "b_min b_max th_min th_max precision_mm" << std::endl;
        std::cerr << "  b_min/b_max     - границы baseline, см (например 5 25)" << std::endl;
        std::cerr << "  th_min/th_max   - границы угла лазера, градусы (например -30 0, или -30 30," << std::endl;
        std::cerr << "                    если знак угла заранее не известен - проверьте на своих данных)" << std::endl;
        std::cerr << "  precision_mm    - требуемая точность по baseline, мм (например 0.001)" << std::endl;
        exit(1);
    }
 
    std::string camArr = argv[1];
    std::string pointsArr = argv[2];
    std::string outputArch = argv[3];
    double bMin = atof(argv[4]);
    double bMax = atof(argv[5]);
    double thMin = atof(argv[6]);
    double thMax = atof(argv[7]);
    double precisionMm = atof(argv[8]);
 
    std::cout << "camArr = " << camArr << std::endl;
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
 
    std::cout << "cx = " << cx << std::endl;
    std::cout << "fx = " << fx << std::endl;
 
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
    std::cout << "Baseline: from " << bMin << " to " << bMax << " cm" << std::endl;
    std::cout << "Theta: from " << thMin << " to " << thMax << " deg" << std::endl;
    std::cout << "Precision: " << precisionMm << " mm" << std::endl;
 
    const double bTol = precisionMm / 10.0;
    const double thTol = 1e-4;
 
    const int GRID_TH = 4;
    const int GRID_B  = 4;
 
    std::vector<Result> starts;
    starts.reserve(GRID_TH * GRID_B);
    for (int i = 0; i < GRID_TH; i++) {
        for (int j = 0; j < GRID_B; j++) {
            double th0 = thMin + (thMax - thMin) * (i + 0.5) / GRID_TH;
            double b0  = bMin  + (bMax  - bMin)  * (j + 0.5) / GRID_B;
            starts.push_back({th0, b0, 0.0});
        }
    }
 
    const double thStep0 = (thMax - thMin) / GRID_TH / 2.0;
    const double bStep0  = (bMax  - bMin)  / GRID_B  / 2.0;
 
    std::vector<Result> results(starts.size());
 
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < starts.size(); i++) {
        results[i] = patternSearch(starts[i].theta, starts[i].baseline,
                                    thMin, thMax, bMin, bMax,
                                    thStep0, bStep0, thTol, bTol);
    }
 
    Result best = results[0];
    for (const auto &r : results) {
        if (r.error < best.error) {
            best = r;
        }
    }
 
    std::cout << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Optimization done... Results: " << std::endl;
    std::cout << "Baseline: " << best.baseline << std::endl;
    std::cout << "Angle: " << best.theta << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "Optimization score: " << best.error << std::endl;
 
    std::vector<double> ba = {best.baseline, best.theta};
    cnpy::npz_save(outputArch,"ba",&ba[0],{2},"w");
 
    std::cout << "Results saved to: " << outputArch << std::endl;
    return 0;
}