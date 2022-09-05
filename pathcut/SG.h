// #pragma GCC optimize("O0")
#pragma once
#include <mitsuba/core/vector.h>
#include <mitsuba/render/bsdf.h>

using namespace mitsuba;
struct SG {
    // static Spectrum piecewiseLinearApproximation() {
    // }
    static SG product(const SG &g1, const SG &g2) {
        float lambda;
        Vector3 i_mean = weightedSumP(g1, g2, lambda);
        Spectrum c = g1.c * g2.c * std::max(math::fastexp(lambda - g1.lambda - g2.lambda), 0.f);
        return SG(i_mean, lambda, c);
    }
    static Vector3 weightedSumP(const SG &g1, const SG &g2, float &length) {
        Vector3 unnormalized = g1.lambda * g1.p + g2.lambda * g2.p;
        length = unnormalized.length();
        return normalize(unnormalized);
    }
    // static Vector3 weightedSumP(const SG &g1, const SG &g2, const SG &g3, float &length) {
    //     Vector3 unnormalized = g1.lambda * g1.p + g2.lambda * g2.p + g3.lambda * g3.p;
    //     length = unnormalized.length();
    //     assert(length > 0.f);
    //     return normalize(unnormalized);
    // }
    static SG sgLight(float radius, const Spectrum &intensity, const Vector3 &dir_unnromalized) {
        float length = dir_unnromalized.length();
        return SG(normalize(dir_unnromalized), 4 * pow(length / radius, 2), 2 * intensity);
    }
    static SG sgDistantLight(const Vector3 &dir, float solidAngle, const Spectrum &intensity) {
        return SG(dir, 4 * M_PI / solidAngle, 2 * intensity);
    }
    static SG brdfSlice(const Normal &n, const Vector3 &o, float roughness, const BSDF *bsdf) {
        float cosThetaO = dot(o, n);
        // if (cosThetaO < 0.f) {
        //     return SG(Vector3(0.f),0.f,Spectrum(0.f));
        // }
        // const SG& ndf, float Mo
        float m2 = roughness * roughness;
        SG ndf(n, 2.f / m2, Spectrum(1.f) / (M_PI * m2));
        float lambda = ndf.lambda / (4.f * cosThetaO);
        Vector3 p = reflect(o, n);
        if (bsdf->isConductor()) {
            Spectrum F = fresnelConductorApprox(cosThetaO, bsdf->getConductorEta(), bsdf->getConductorK());
            float G1 = 1.0f / (cosThetaO + sqrt(m2 + (1 - m2) * cosThetaO * cosThetaO));
            return SG(p, lambda, F * G1 * G1 * ndf.c);
        } else {
            float eta = bsdf->getProperties().getFloat("intIOR");
            Spectrum d = bsdf->getProperties().getSpectrum("diffuseReflectance");
            Spectrum F(fresnelDielectricExt(cosThetaO, eta));
            float G1 = 1.0f / (cosThetaO + sqrt(m2 + (1 - m2) * cosThetaO * cosThetaO));
            return SG(p, lambda, F * G1 * G1 * ndf.c * d);
        }
    }
    static SG btdfSlice(const SG &ndf, const Spectrum &mo, const Vector3 &o, float _eta) {
        // eta: intIOR/extIOR
        float cosThetaO = dot(o, ndf.p);
        Vector3 p = refract(o, ndf.p, _eta, cosThetaO);
        float eta = cosThetaO < 0 ? _eta : (1.f / _eta);
        float sqrtDenom = cosThetaO + eta * cosThetaO;
        float dwh_dwo = (eta * eta * cosThetaO) / (sqrtDenom * sqrtDenom);
        float lambda = ndf.lambda * dwh_dwo;
        return SG(p, lambda, mo * ndf.c);
    }
    static Spectrum convolve(const SG &g1, const SG &g2) {
        float dm = (g1.lambda * g1.p + g2.lambda * g2.p).length();
        Spectrum expo = math::fastexp(dm - g1.lambda - g2.lambda) * g1.c * g2.c;
        float other = 1.0f - math::fastexp(-2.0f * dm);
        return (2.0f * M_PI * expo * other) / dm;
    }
    static SG convolveApproximation(const SG &gli, const SG &gs, const Normal &n) {
        float lambda3;
        Vector3 i_mean = weightedSumP(gli, gs, lambda3);
        Spectrum c = 2 * M_PI * gli.c * gs.c * std::fabs(dot(i_mean, n)) / lambda3;
        float lambda = gli.lambda * gs.lambda / (gli.lambda + gs.lambda);
        return SG(reflect(gli.p, n), lambda, c);
    }
    static SG ndf(float roughness, const Normal &n) {
        float m2 = roughness * roughness;
        return SG(n, 2.f / m2, Spectrum(1.f) / (M_PI * m2));
    }
    static SG region(const Point &pos, const Point &center, const Vector3 &normal, float area) {
        Vector3 dir_unnormalized = center - pos;
        Vector3 p = normalize(dir_unnormalized);
        float omega = area * fabs(dot(p, normal)) / pow(dir_unnormalized.length(), 2);
        return SG(p, 4 * M_PI / omega, Spectrum(2));
    }
    static Vector3 frameToWorld(const Vector3 n, const Vector3 dir) {
        Vector3 s, t;
        if (std::abs(n.x) > std::abs(n.y)) {
            mitsuba::Float invLen = 1.0f / std::sqrt(n.x * n.x + n.z * n.z);
            t = Vector3(n.z * invLen, 0.0f, -n.x * invLen);
        } else {
            mitsuba::Float invLen = 1.0f / std::sqrt(n.y * n.y + n.z * n.z);
            t = Vector3(0.0f, n.z * invLen, -n.y * invLen);
        }
        s = cross(t, n);
        return s * dir.x + t * dir.y + n * dir.z;
    }

    SG() {}
    SG(const Vector3 &p, float lambda, const Spectrum &c) : p(p), c(c), lambda(lambda) {
    }
    void computeNormalization() {
        eMin2Lambda = math::fastexp(-2 * lambda);
        norm = lambda / (2 * M_PI * (1 - eMin2Lambda));
    }
    Spectrum evaluate(const Vector3 &x) const {
        return c * math::fastexp(lambda * (dot(x, p) - 1));
    }
    Spectrum evaluate(const Vector3 &center, const Vector3 &x) const {
        return c * math::fastexp(lambda * (dot(x, center) - 1));
    }
    Spectrum integral() const {
        return 2 * M_PI * c * (1 - eMin2Lambda) / lambda;
    }

    float pdf(const Vector3 &x) const {
        return math::fastexp(lambda * (dot(x, p) - 1)) * norm;
    }

    float pdf(const Vector3 &center, const Vector3 &x) const {
        float dotVal = dot(x, center);
        return math::fastexp(lambda * (dotVal - 1.f)) * norm;
    }

    Vector3 sample(const Point2 sample) const {
        float cosTheta = 1 + (std::log(sample.x + eMin2Lambda * (1 - sample.x))) / lambda;
        const float sinTheta = (1.0f - cosTheta * cosTheta <= 0.0f) ? 0.0f : std::sqrt(1.f - cosTheta * cosTheta);
        const float phi = 2.f * M_PI * sample[1];

        float sinPhi, cosPhi;
        sincosf(phi, &sinPhi, &cosPhi);
        Vector3 dir(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        dir = frameToWorld(p, dir);
        return dir;
    }

    Vector3 sample(const Vector3 &center, const Point2 sample) const {
        const float cosTheta = 1.f + std::log1p(eMin2Lambda * sample.x - sample.x) / lambda;
        // float cosTheta = 1 + (std::log(sample.x + eMin2Lambda * (1 - sample.x))) / lambda;
        const float sinTheta = (1.0f - cosTheta * cosTheta <= 0.0f) ? 0.0f : std::sqrt(1.f - cosTheta * cosTheta);
        const float phi = 2.f * M_PI * sample[1];

        float sinPhi, cosPhi;
        sincosf(phi, &sinPhi, &cosPhi);
        Vector3 dir(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        dir = frameToWorld(center, dir);
        return dir;
    }

    Vector3 p;
    Spectrum c;
    float lambda;
    float eMin2Lambda;
    float norm;
};
