#include "newton.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/src/Geometry/OrthoMethods.h>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

using autodiff::ArrayXreal;
using autodiff::real;
using autodiff::Vector3real;
using autodiff::VectorXreal;

#define MAX_STEP 20

float solveOneBounce(VertInfo &reflVert, const std::array<VertInfo, 3> &triInfo,
                     const std::array<float, 3> &lightPos_, const std::array<float, 3> &camPos_, float errorThreshold) {
    float minError = INFINITY;
    Vector3real lightPos(lightPos_[0], lightPos_[1], lightPos_[2]);
    Vector3real camPos(camPos_[0], camPos_[1], camPos_[2]);
    Vector3real pos;
    Vector3real normal;

    auto f = [lightPos, camPos, triInfo, &pos, &normal](const VectorXreal &x) {
        auto alpha = x[0];
        auto beta = x[1];
        auto gamma = 1 - alpha - beta;

        pos[0] = triInfo[0].pos[0] * alpha + triInfo[1].pos[0] * beta + triInfo[2].pos[0] * gamma;
        pos[1] = triInfo[0].pos[1] * alpha + triInfo[1].pos[1] * beta + triInfo[2].pos[1] * gamma;
        pos[2] = triInfo[0].pos[2] * alpha + triInfo[1].pos[2] * beta + triInfo[2].pos[2] * gamma;
        normal[0] = triInfo[0].normal[0] * alpha + triInfo[1].normal[0] * beta + triInfo[2].normal[0] * gamma;
        normal[1] = triInfo[0].normal[1] * alpha + triInfo[1].normal[1] * beta + triInfo[2].normal[1] * gamma;
        normal[2] = triInfo[0].normal[2] * alpha + triInfo[1].normal[2] * beta + triInfo[2].normal[2] * gamma;
        normal.normalize();

        Vector3real wi, wo;
        wi = (lightPos - pos).normalized();
        wo = (camPos - pos).normalized();
        Vector3real h = (wi + wo).normalized();
        Vector3real result = normal - h;
        return result;
    };

    VectorXreal x(2);
    x.fill(1.0 / 3.0);
    for (int iter = 0; iter < MAX_STEP; iter++) {
        VectorXreal F;

        auto J = autodiff::jacobian(f, autodiff::wrt(x), autodiff::at(x), F);
        float error = F.cwiseAbs().sum().val();
        // error ?
        if (error < minError) {
            minError = error;
            reflVert.pos[0] = pos[0].val();
            reflVert.pos[1] = pos[1].val();
            reflVert.pos[2] = pos[2].val();
            reflVert.normal[0] = normal[0].val();
            reflVert.normal[1] = normal[1].val();
            reflVert.normal[2] = normal[2].val();
            if (error < errorThreshold) {
                return error;
            }
        }

        auto JT = J.transpose();
        auto invJ = (JT * J).inverse() * JT;
        auto x_delta = invJ * F;
        auto x_delta2 = x_delta / 4.0;
        x = x - x_delta2;
        float bound = 0.2;
        if (x[0] < 0 - bound || x[0] > 1 + bound || x[1] < 0 - bound || x[1] > 1 + bound || x[0] + x[1] < 0 - bound || x[0] + x[1] > 1 + bound) {
            auto edge = [=](int idx1, int idx2) {
                return Vector3real(triInfo[idx1].pos[0] - triInfo[idx2].pos[0], triInfo[idx1].pos[1] - triInfo[idx2].pos[1], triInfo[idx1].pos[2] - triInfo[idx2].pos[2]);
            };
            auto pos = [=](int idx) {
                return Vector3real(triInfo[idx].pos[0], triInfo[idx].pos[1], triInfo[idx].pos[2]);
            };
            Vector3real p;
            for (int dim = 0; dim < 3; dim++) {
                p[dim] = x[0] * triInfo[0].pos[dim] + x[1] * triInfo[1].pos[dim] + (1 - x[0] - x[1]) * triInfo[2].pos[dim];
            }
            if (x[0] < 0) {
                auto e = edge(2, 1);
                real t = (p - pos(1)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[0] = 0;
                x[1] = 1 - t;
            } else if (x[1] < 0) {
                auto e = edge(0, 2);
                real t = (p - pos(2)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[0] = t;
                x[1] = 0;
            } else if (x[0] + x[1] > 1) {
                auto e = edge(1, 0);
                real t = (p - pos(0)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[0] = 1 - t;
                x[1] = t;
            }
        }

        // auto maxCoeff = x_delta.maxCoeff();
        // auto minCoeff = x_delta.minCoeff();
        // x = x - x_delta / std::max(1.0, std::max(fabs(minCoeff.val()), maxCoeff.val()));
        // std::cout << "x_delta: \n"
        //           << x_delta << "\n";
        // std::cout << "x_delta2: \n"
        //           << x_delta2 << "\n";
        // std::cout << "F: \n"
        //           << F << "\n";
        // std::cout << "error: \n"
        //           << error << "\n";
        // std::cout << "vertPos: \n"
        //           << pos << "\n";
        // std::cout << "vertN: \n"
        //           << normal << "\n";
        // std::cin.get();
    }
    return minError;
}

float solveTwoBounce(VertInfo &reflVert1, VertInfo &reflVert2, const std::array<VertInfo, 3> &triInfo1, const std::array<VertInfo, 3> &triInfo2,
                     const std::array<float, 3> &lightPos_, const std::array<float, 3> &camPos_, float errorThreshold) {
    float minError = INFINITY;
    Vector3real lightPos(lightPos_[0], lightPos_[1], lightPos_[2]);
    Vector3real camPos(camPos_[0], camPos_[1], camPos_[2]);
    Vector3real pos1, pos2;
    Vector3real n1, n2;
    auto f = [lightPos, camPos, triInfo1, triInfo2, &pos1, &n1, &pos2, &n2](const VectorXreal &x) {
        // first point
        auto alpha1 = x[0];
        auto beta1 = x[1];
        auto gamma1 = 1 - alpha1 - beta1;
        pos1[0] = triInfo1[0].pos[0] * alpha1 + triInfo1[1].pos[0] * beta1 + triInfo1[2].pos[0] * gamma1;
        pos1[1] = triInfo1[0].pos[1] * alpha1 + triInfo1[1].pos[1] * beta1 + triInfo1[2].pos[1] * gamma1;
        pos1[2] = triInfo1[0].pos[2] * alpha1 + triInfo1[1].pos[2] * beta1 + triInfo1[2].pos[2] * gamma1;
        n1[0] = triInfo1[0].normal[0] * alpha1 + triInfo1[1].normal[0] * beta1 + triInfo1[2].normal[0] * gamma1;
        n1[1] = triInfo1[0].normal[1] * alpha1 + triInfo1[1].normal[1] * beta1 + triInfo1[2].normal[1] * gamma1;
        n1[2] = triInfo1[0].normal[2] * alpha1 + triInfo1[1].normal[2] * beta1 + triInfo1[2].normal[2] * gamma1;
        n1.normalize();
        // second point
        auto alpha2 = x[2];
        auto beta2 = x[3];
        auto gamma2 = 1 - alpha2 - beta2;
        pos2[0] = triInfo2[0].pos[0] * alpha2 + triInfo2[1].pos[0] * beta2 + triInfo2[2].pos[0] * gamma2;
        pos2[1] = triInfo2[0].pos[1] * alpha2 + triInfo2[1].pos[1] * beta2 + triInfo2[2].pos[1] * gamma2;
        pos2[2] = triInfo2[0].pos[2] * alpha2 + triInfo2[1].pos[2] * beta2 + triInfo2[2].pos[2] * gamma2;
        n2[0] = triInfo2[0].normal[0] * alpha2 + triInfo2[1].normal[0] * beta2 + triInfo2[2].normal[0] * gamma2;
        n2[1] = triInfo2[0].normal[1] * alpha2 + triInfo2[1].normal[1] * beta2 + triInfo2[2].normal[1] * gamma2;
        n2[2] = triInfo2[0].normal[2] * alpha2 + triInfo2[1].normal[2] * beta2 + triInfo2[2].normal[2] * gamma2;
        n2.normalize();

        Vector3real wi1, wo1, h1, wi2, wo2, h2;
        wi1 = (lightPos - pos1).normalized();
        wo1 = (pos2 - pos1).normalized();
        h1 = (wi1 + wo1).normalized();

        wi2 = (pos1 - pos2).normalized();
        wo2 = (camPos - pos2).normalized();
        h2 = (wi2 + wo2).normalized();
        Vector3real d1 = n1 - h1;
        Vector3real d2 = n2 - h2;
        VectorXreal result(6);
        result[0] = d1[0];
        result[1] = d1[1];
        result[2] = d1[2];
        result[3] = d2[0];
        result[4] = d2[1];
        result[5] = d2[2];
        return result;
    };

    VectorXreal x(4);
    x.fill(1.0 / 3.0);

    for (int iter = 0; iter < MAX_STEP; iter++) {

        VectorXreal F;

        auto J = autodiff::jacobian(f, autodiff::wrt(x), autodiff::at(x), F);
        float error = F.cwiseAbs().sum().val();
        if (error < minError) {
            minError = error;
            reflVert1.pos[0] = pos1[0].val();
            reflVert1.pos[1] = pos1[1].val();
            reflVert1.pos[2] = pos1[2].val();
            reflVert1.normal[0] = n1[0].val();
            reflVert1.normal[1] = n1[1].val();
            reflVert1.normal[2] = n1[2].val();
            reflVert2.pos[0] = pos2[0].val();
            reflVert2.pos[1] = pos2[1].val();
            reflVert2.pos[2] = pos2[2].val();
            reflVert2.normal[0] = n2[0].val();
            reflVert2.normal[1] = n2[1].val();
            reflVert2.normal[2] = n2[2].val();
            if (error < errorThreshold) {
                return error;
            }
        }

        auto JT = J.transpose();
        auto invJ = (JT * J).inverse() * JT;
        auto x_delta = invJ * F;
        auto x_delta2 = x_delta / 4.0;
        x = x - x_delta2;
        float bound = 0;
        if (x[0] < 0 - bound || x[0] > 1 + bound || x[1] < 0 - bound || x[1] > 1 + bound || x[0] + x[1] < 0 - bound || x[0] + x[1] > 1 + bound) {
            auto edge = [=](int idx1, int idx2) {
                return Vector3real(triInfo1[idx1].pos[0] - triInfo1[idx2].pos[0], triInfo1[idx1].pos[1] - triInfo1[idx2].pos[1], triInfo1[idx1].pos[2] - triInfo1[idx2].pos[2]);
            };
            auto pos = [=](int idx) {
                return Vector3real(triInfo1[idx].pos[0], triInfo1[idx].pos[1], triInfo1[idx].pos[2]);
            };
            Vector3real p;
            for (int dim = 0; dim < 3; dim++) {
                p[dim] = x[0] * triInfo1[0].pos[dim] + x[1] * triInfo1[1].pos[dim] + (1 - x[0] - x[1]) * triInfo1[2].pos[dim];
            }
            if (x[0] < 0) {
                auto e = edge(2, 1);
                real t = (p - pos(1)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[0] = 0;
                x[1] = 1 - t;
            } else if (x[1] < 0) {
                auto e = edge(0, 2);
                real t = (p - pos(2)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[0] = t;
                x[1] = 0;
            } else if (x[0] + x[1] > 1) {
                auto e = edge(1, 0);
                real t = (p - pos(0)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[0] = 1 - t;
                x[1] = t;
            }
            iter = std::max(18, iter);
        }
        if (x[2] < 0 - bound || x[2] > 1 + bound || x[3] < 0 - bound || x[3] > 1 + bound || x[2] + x[3] < 0 - bound || x[2] + x[3] > 1 + bound) {
            auto edge = [=](int idx1, int idx2) {
                return Vector3real(triInfo2[idx1].pos[0] - triInfo2[idx2].pos[0], triInfo2[idx1].pos[1] - triInfo2[idx2].pos[1], triInfo2[idx1].pos[2] - triInfo2[idx2].pos[2]);
            };
            auto pos = [=](int idx) {
                return Vector3real(triInfo2[idx].pos[0], triInfo2[idx].pos[1], triInfo2[idx].pos[2]);
            };
            Vector3real p;
            for (int dim = 0; dim < 3; dim++) {
                p[dim] = x[2] * triInfo2[0].pos[dim] + x[3] * triInfo2[1].pos[dim] + (1 - x[2] - x[3]) * triInfo2[2].pos[dim];
            }
            if (x[2] < 0) {
                auto e = edge(2, 1);
                real t = (p - pos(1)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[2] = 0;
                x[3] = 1 - t;
            } else if (x[3] < 0) {
                auto e = edge(0, 2);
                real t = (p - pos(2)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[2] = t;
                x[3] = 0;
            } else if (x[2] + x[3] > 1) {
                auto e = edge(1, 0);
                real t = (p - pos(0)).dot(e) / e.squaredNorm();
                t = std::min(1.0, std::max(t.val(), 0.0));
                x[2] = 1 - t;
                x[3] = t;
            }
        }
    }
    return minError;
}