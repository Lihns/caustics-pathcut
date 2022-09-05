// #define MYDEBUG
#ifdef MYDEBUG
#pragma GCC optimize("O0")
#endif

#pragma once
#include <omp.h>
#include <queue>
#include <vector>

#include "SG.h"
#include "newton.h"
#include "ptree.h"

using namespace mitsuba;

struct IntervalPath {
    // static PTree *ptree;
    struct Vert {
        const PNode *pnode;
        const TriMesh *mesh;
        int type;
        bool guided;
    };
    std::vector<Vert> bounces;
    Point lightPos;
    Point camPos;
    Vector camDir;
    Spectrum lightIntensity;
    Interval3D lightBound;
    Interval3D lightDirBound;
    int lightType;
    float lightCutoff;
};

struct ParallaxStruct {
    Frame reflFrame;
    Point origin;
    float diU, hiU, diV, hiV;
    ParallaxStruct() : diU(INFINITY), hiU(0), diV(INFINITY), hiV(0) {}
    ParallaxStruct(const Point &reflPoint, const Point &lightPos, const Frame &tangentFrame, float focalLengthU, float focalLengthV,
                   float &d_o, float &hoU, float &hoV)
        : reflFrame(tangentFrame), origin(reflPoint) {
        Vector lightRel = tangentFrame.toLocal(lightPos - reflPoint);
        hoU = lightRel.x;
        hoV = lightRel.y;
        d_o = lightRel.z;
        diU = 1 / (1 / focalLengthU - 1 / d_o),
        diV = 1 / (1 / focalLengthV - 1 / d_o),
        hiU = -diU / d_o * hoU,
        hiV = -diV / d_o * hoV;
    }
    void parallaxDir(const Point &pos, Vector &dir) const {
        if (std::isinf(diU) && std::isinf(diV)) {
            return;
        }
        Vector posRel = pos - origin;
        Vector posLocal = reflFrame.toLocal(posRel);
        float de = posLocal.z, heU = posLocal.x, heV = posLocal.y;
        float deltaU = (diU * heU - de * hiU) / (diU - de);
        float deltaV = (diV * heV - de * hiV) / (diV - de);
        Vector reflPoint = reflFrame.s * deltaU + reflFrame.t * deltaV;
        dir = normalize(reflPoint - posRel);
    }
    void imageDist(const Point &pos, float &distU, float &distV) const {
        if (std::isinf(diU) && std::isinf(diV)) {
            return;
        }
        Vector posRel = pos - origin;
        Vector posLocal = reflFrame.toLocal(posRel);
        float de = posLocal.z, heU = posLocal.x, heV = posLocal.y;
        distU = std::sqrt((de - diU) * (de - diU) + (heU - hiU) * (heU - hiU)) / std::sqrt(diU * diU + hiU * hiU);
        distV = std::sqrt((de - diV) * (de - diV) + (heV - hiV) * (heV - hiV)) / std::sqrt(diV * diV + hiV * hiV);
    }
    Point imageWorldPosU() const {
        Vector imageRel = diU * reflFrame.n + hiU * reflFrame.s;
        Point world = origin + imageRel;
        return world;
    }
    Point imageWorldPosV() const {
        Vector imageRel = diV * reflFrame.n + hiV * reflFrame.t;
        Point world = origin + imageRel;
        return world;
    }
};
struct SGLight {
    SG sg;
    ParallaxStruct parallaxInfo;
    float weight;
};

struct SGMixture {
    std::vector<SGLight> sgLights;

    Spectrum eval(Vector3 d) const {
        Spectrum result(0.f);
        for (size_t idx = 0; idx < sgLights.size(); idx++) {
            result += sgLights[idx].sg.evaluate(d);
        }
        return result;
    }
    Spectrum evalParallax(Point pos, Vector3 d) const {
        Spectrum result(0.f);
        for (const auto &vert : sgLights) {
            Vector lobeCenter = vert.sg.p;
            vert.parallaxInfo.parallaxDir(pos, lobeCenter);
            result += vert.sg.evaluate(lobeCenter, d);
            // result += vert.sg.evaluate(d);
        }
        return result;
    }
    Spectrum evalProduct(const Intersection &its, const Vector &dir) {
        const auto &bsdf = its.getBSDF();
        Vector wi = its.toWorld(its.wi);
        SG brdfLobe = SG::brdfSlice(its.shFrame.n, wi, bsdf->getRoughness(its, 0), bsdf);

        Spectrum result;
        for (const auto &sgLight : sgLights) {
            SG incidentLobe = sgLight.sg;
            sgLight.parallaxInfo.parallaxDir(its.p, incidentLobe.p);
            SG product = SG::product(incidentLobe, brdfLobe);
            result += product.evaluate(dir);
        }
        return result;
    }

    int sgNum() const {
        return sgLights.size();
    }

    void merge(float mergeAngleThreshold, float mergeDistThreshold) {
        if (mergeAngleThreshold == 0 && mergeDistThreshold == 0) {
            return;
        }
        std::vector<SGLight> mergedSGs;
        while (!sgLights.empty()) {
            size_t idx = 1;
            mergedSGs.push_back(sgLights[idx]);
            sgLights[1] = sgLights.back();
            sgLights.pop_back();
            while (idx < sgLights.size()) {
                auto &merged = mergedSGs.back();
                const auto &vert = sgLights[idx];

                // float cosine = dot(merged.sg.p, vert.sg.p);
                float dist = (merged.parallaxInfo.origin - vert.parallaxInfo.origin).length();

                if (dist < mergeDistThreshold) {
                    // float cosine = dot(merged.sg.p, vert.sg.p);
                    // Vector p = normalize(merged.sg.p + vert.sg.p);
                    // float lambda = std::min(merged.sg.lambda, 1 / (1 - std::cos((std::acos(cosine) + std::acos(-1 / merged.sg.lambda + 1) + std::acos(-1 / vert.sg.lambda + 1)) / 2)));
                    // Spectrum c = (merged.sg.c / merged.sg.lambda + vert.sg.c / vert.sg.lambda) * lambda;
                    merged.sg.c += vert.sg.c;
                    merged.sg.lambda = std::min(merged.sg.lambda, vert.sg.lambda);
                    sgLights[idx] = sgLights.back();
                    sgLights.pop_back();
                } else {
                    idx++;
                }
            }
        }
        sgLights.swap(mergedSGs);
    }
    void normalizeForSampling() {
        for (size_t i = 0; i < sgLights.size(); i++) {
            sgLights[i].sg.computeNormalization();
        }
    }
    void computeWeights() __attribute__((optimize("O3"))) { //????
        if (sgLights.empty()) {
            return;
        }
        float sgWeightSum = 0.f;
        for (size_t i = 1; i < sgLights.size(); i++) {
            float weight = sgLights[i].sg.integral().getLuminance();
            sgLights[i].weight = weight;
            sgWeightSum += weight;
        }
        sgLights[0].weight = (sgWeightSum == 0.f ? 1 : sgWeightSum) / sgLights.size();
        sgWeightSum += sgLights[0].weight;
        for (size_t i = 0; i < sgLights.size(); i++) {
            sgLights[i].weight /= sgWeightSum;
        }
    }

    Vector sampleWeighted(const Point &pos, Point2 sample) const {
        float partition = sample.x;
        float sum = 0;
        for (size_t i = 0; i < sgLights.size(); i++) {
            if (sum + sgLights[i].weight > partition) {
                sample.x = (partition - sum) / (sgLights[i].weight);
                Vector lobeCenter = sgLights[i].sg.p;
                sgLights[i].parallaxInfo.parallaxDir(pos, lobeCenter);
                Vector wo = sgLights[i].sg.sample(lobeCenter, sample);
                return wo;
            }
            sum = sum + sgLights[i].weight;
        }
        return Vector(0.f);
    }
    float pdfWeighted(const Point &pos, const Vector &dir) const {
        float pdf = 0.f;
        for (size_t i = 0; i < sgLights.size(); i++) {
            Vector lobeCenter = sgLights[i].sg.p;
            sgLights[i].parallaxInfo.parallaxDir(pos, lobeCenter);
            pdf += sgLights[i].weight * sgLights[i].sg.pdf(lobeCenter, dir);
        }
        return pdf;
    }
    Vector sampleProduct(const Intersection &its, Point2 sample) const {
        sample.x *= sgLights.size();
        int idx = int(sample.x);
        sample.x -= idx;
        const auto &sampleSGLight = sgLights[idx];

        SG incidentLobe = sampleSGLight.sg;
        sampleSGLight.parallaxInfo.parallaxDir(its.p, incidentLobe.p);

        const auto &bsdf = its.getBSDF();
        Vector wi = its.toWorld(its.wi);
        SG brdfLobe = SG::brdfSlice(its.shFrame.n, wi, bsdf->getRoughness(its, 0), bsdf);
        SG product = SG::product(incidentLobe, brdfLobe);
        product.computeNormalization();
        Vector dir = product.sample(sample);
        return dir;
    }
    float pdfProduct(const Intersection &its, const Vector &dir) const {
        const auto &bsdf = its.getBSDF();
        Vector wi = its.toWorld(its.wi);
        SG brdfLobe = SG::brdfSlice(its.shFrame.n, wi, bsdf->getRoughness(its, 0), bsdf);

        float pdf = 0.f;
        for (const auto &sgLight : sgLights) {
            SG incidentLobe = sgLight.sg;
            sgLight.parallaxInfo.parallaxDir(its.p, incidentLobe.p);
            SG product = SG::product(incidentLobe, brdfLobe);
            product.computeNormalization();
            pdf += product.pdf(dir);
        }
        return pdf / sgLights.size();
    }
};

struct SGShape {
    std::vector<SGMixture> sgMixtures;
};

/*****************************************************************************/
/********************************   pathcut   ********************************/
/*****************************************************************************/
class Pathcut {
private:
    // std::vector<SGLight> m_sgLights;
    int m_nBounce;
    const Scene *m_scene;
    omp_lock_t m_lock;

    float m_pathcutCount;
    float m_validPathcutCount;
    float m_sgCount;

    bool m_validateCamera;
    bool m_parallax;
    bool m_newton;

public:
    Pathcut(std::vector<SGShape> &sgShapes, const Scene *scene, const std::vector<PTree> &ptrees, const Point &camPos, const Vector &camDir, int bounce,
            float mergeAngleThreshold, float mergeDistThreshold,
            bool sds = false, bool parallax = true, bool newton = true) {
        m_scene = scene;
        m_nBounce = bounce;
        m_validateCamera = !sds;
        m_parallax = parallax;
        m_newton = newton;

        m_pathcutCount = 0;
        m_validPathcutCount = 0;
        m_sgCount = 0;
        // find available pathcuts, specular paths and SGs.
        ref<Timer> timer = new Timer();
        computePathcuts(sgShapes, ptrees, camPos, camDir, mergeAngleThreshold, mergeDistThreshold);
        std::cout << "pathcut detph: " << m_nBounce << ", done in " << timer->getSeconds() << "s\n";
        std::cout << "leaf pathcut found: " << m_pathcutCount << "\n";
        std::cout << "valid pathcut: " << m_validPathcutCount << "\n";
        std::cout << "valid SG: " << m_sgCount << "\n";
    }

private:
    std::vector<IntervalPath> emitterPaths(const std::vector<PTree> &ptrees, const Point &camPos, const Vector &camDir) const {
        std::vector<IntervalPath> roots;

        const auto &emitters = m_scene->getEmitters();
        for (const auto &emitter : emitters) {
            if (!emitter->getProperties().getBoolean("causticEmitter", false)) {
                continue;
            }
            IntervalPath rootPath;
            rootPath.camPos = camPos;
            rootPath.camDir = camDir;
            if (emitter->getProperties().getString("lightType") == "point") {
                rootPath.lightPos = emitter->getProperties().getPoint("position");
                rootPath.lightIntensity = emitter->getProperties().getSpectrum("intensity");
                rootPath.lightType = 0;
                rootPath.lightBound = Interval3D(rootPath.lightPos);
                rootPath.lightDirBound = Interval3D(Point(-1.f), Point(1.f));
                std::cout << "construct root path cut for point light.\n";
            } else if (emitter->getProperties().getString("lightType") == "area") {
                // expand normal bound of area light
                auto mesh = const_cast<Shape *>(emitter->getShape())->createTriMesh();
                const auto &normals = mesh->getVertexNormals();
                auto vertexCount = mesh->getVertexCount();
                for (size_t i = 0; i < vertexCount; i++)
                    rootPath.lightDirBound.expand(normals[i]);
                // expand pos bound of area light
                const auto aabb = emitter->getShape()->getAABB();
                rootPath.lightPos = aabb.getCenter();
                rootPath.lightIntensity = emitter->getProperties().getSpectrum("radiance");
                rootPath.lightType = 1;
                rootPath.lightBound = Interval3D(aabb.min, aabb.max);
                std::cout << "construct root path cut for area light.\n";
            } else if (emitter->getProperties().getString("lightType") == "directional") {
                const auto dir = emitter->getProperties().getVector("direction");
                rootPath.lightIntensity = emitter->getProperties().getSpectrum("irradiance");
                rootPath.lightType = 2;
                rootPath.lightBound = Interval3D(0.f);
                rootPath.lightDirBound = Interval3D(dir);
                std::cout << "construct root path cut for directional light.\n";
            } else if (emitter->getProperties().getString("lightType") == "spot") {
                Transform trafo = emitter->getWorldTransform()->eval(0.f);
                rootPath.lightPos = trafo(Point(0.f));
                rootPath.lightIntensity = emitter->getProperties().getSpectrum("intensity");
                rootPath.lightType = 3;
                rootPath.lightBound = Interval3D(rootPath.lightPos);
                rootPath.lightDirBound = Interval3D(trafo(Vector(0, 0, 1)));
                rootPath.lightCutoff = degToRad(emitter->getProperties().getFloat("cutoffAngle", 20));
                std::cout << "construct root path cut for spot light.\n";
            } else {
                std::cout << "unsupported light type.\n";
                continue;
            }
            constructRoot(rootPath, ptrees, roots, 0);
        }
        return roots;
    }
    void computePathcuts(std::vector<SGShape> &sgShapes, const std::vector<PTree> &ptrees, const Point &camPos, const Vector &camDir, float mergeAngleThreshold, float mergeDistThreshold) {
        // root path
        auto roots = emitterPaths(ptrees, camPos, camDir);
        std::cout << "root pathcuts: " << roots.size() << std::endl;

        omp_init_lock(&m_lock);
        for (size_t i = 0; i < roots.size(); i++) {
            if (validateReflection(roots[i])) {
                std::deque<IntervalPath> paths;
                paths.push_back(roots[i]);
                // subdivide several times
                while (paths.size() < 1000 && !paths.empty()) {
                    const IntervalPath &path = paths.front();
                    if (validateReflection(path)) {
                        float thickness;
                        int idx = findThickest(path, thickness);
                        if (idx == -1)
                            break;
                        subdivide(paths, path, idx);
                    }
                    paths.pop_front();
                }
#ifndef MYDEBUG
#pragma omp parallel for schedule(dynamic)
#endif
                for (size_t idx = 0; idx < paths.size(); idx++) {
                    findPathcuts(sgShapes, paths[idx]);
                }

                int mergedSGCount = 0;

                for (auto &shape : sgShapes) {
#ifndef MYDEBUG
#pragma omp parallel for schedule(dynamic)
#endif
                    for (size_t idx = 0; idx < shape.sgMixtures.size(); idx++) {
                        shape.sgMixtures[idx].merge(mergeAngleThreshold, mergeDistThreshold);
                        omp_set_lock(&m_lock);
                        mergedSGCount += shape.sgMixtures[idx].sgLights.size();
                        omp_unset_lock(&m_lock);
                    }
                }
                std::cout << "merged sg count: " << mergedSGCount << "\n";
            }
        }
        omp_destroy_lock(&m_lock);
    }
    void constructRoot(IntervalPath &rootPath, const std::vector<PTree> &ptrees, std::vector<IntervalPath> &roots, int bounce) const {
        if (bounce == m_nBounce) {
            roots.push_back(rootPath);
            return;
        }
        for (size_t i = 0; i < ptrees.size(); i++) {
            PNode *node = ptrees[i].shapeRoot.get();
            if (node == nullptr) {
                continue;
            }
            if (!rootPath.bounces.empty() && rootPath.bounces.back().pnode == node) {
                continue;
            }
            const auto &mesh = ptrees[i].trimesh;
            const auto &meshProps = mesh->getProperties();
            if (bounce == m_nBounce - 1) {
                if (!(meshProps.getBoolean("reflector", false) || meshProps.getBoolean("refractor", false)) && meshProps.getBoolean("guided", false)) {
                    rootPath.bounces.push_back({node, mesh, 0, true});
                    roots.push_back(rootPath);
                    rootPath.bounces.pop_back();
                }
            } else {
                if (meshProps.getBoolean("reflector", false)) {
                    rootPath.bounces.push_back({node, mesh, 1, meshProps.getBoolean("guided", false)});
                    constructRoot(rootPath, ptrees, roots, bounce + 1);
                    rootPath.bounces.pop_back();
                }
                if (meshProps.getBoolean("refractor", false)) {
                    rootPath.bounces.push_back({node, mesh, -1, meshProps.getBoolean("guided", false)});
                    constructRoot(rootPath, ptrees, roots, bounce + 1);
                    rootPath.bounces.pop_back();
                }
            }
        }
    }
    void findPathcuts(std::vector<SGShape> &sgShapes, const IntervalPath &root) {
        std::stack<IntervalPath> pathStack;
        pathStack.push(root);
        // subdivide several times
        while (!pathStack.empty()) {
            IntervalPath path = pathStack.top();
            pathStack.pop();

            if (validateReflection(path)) {
                float thickness;
                auto shouldSplitIdx = findThickest(path, thickness);
                // auto shouldSplitIdx = findShouldSplit(path, 0.5, posExtent, vecExtent);
                if (shouldSplitIdx == -1) {
                    std::vector<Point> points;
                    std::vector<Vector> normals;
                    omp_set_lock(&m_lock);
                    m_pathcutCount++;
                    omp_unset_lock(&m_lock);
                    if (solvePathcut(path, points, normals) && validateVisibility(points)) {
                        omp_set_lock(&m_lock);
                        m_validPathcutCount++;
                        omp_unset_lock(&m_lock);
                        computeSGs(sgShapes, path, points, normals);
                    }
                } else {
                    subdivide(pathStack, path, shouldSplitIdx);
                }
            }
        }
    }
    bool validateRefraction(const IntervalPath &path) const {
        return false;
    }
    bool validateReflection(const IntervalPath &path) const {
        for (size_t i = 0; i < path.bounces.size() - 1; i++) {
            const auto &curr = path.bounces[i].pnode;
            if (curr->tri && curr == path.bounces[i + 1].pnode)
                return false;
        }

        Interval3D curr2pre;
        const auto &firstNode = path.bounces[0].pnode;
        // emitter to first
        curr2pre = path.lightBound - firstNode->posBox;
        if (path.lightType == 1) { // area
            if (path.lightDirBound.dot(-curr2pre).vMax < 0.f) {
                return false;
            }
        } else if (path.lightType == 2) { // directional
            curr2pre = -path.lightDirBound;
        } else if (path.lightType == 3) { // spot
            if (path.lightDirBound.dot(-curr2pre.normalized()).vMax < cos(path.lightCutoff)) {
                return false;
            }
        }
        // validate camera
        Interval3D cam2last;
        if (m_validateCamera) {
            const auto &lastNode = *path.bounces.back().pnode;
            cam2last = lastNode.posBox - path.camPos;
            if (cam2last.dot(path.camDir).vMax < 0.f) {
                return false;
            }
        }
        Interval3D ibox = curr2pre.normalized();
        for (size_t i = 0; i < path.bounces.size(); i++) {
            const auto &node = *path.bounces[i].pnode;
            const auto &currBox = node.posBox;

            Interval3D curr2next;
            if (i == path.bounces.size() - 1) {
                if (m_validateCamera) {
                    curr2next = -cam2last;
                } else {
                    // not validating camera
                    curr2next = Interval3D(Point(-1, -1, -1), Point(1, 1, 1));
                }
            } else {
                const auto &nextNode = *path.bounces[i + 1].pnode;
                curr2next = nextNode.posBox - currBox;
            }
            Interval3D obox = curr2next.normalized();

            Interval1D dotIN = node.norbox.dot(ibox);
            Interval1D dotON = node.norbox.dot(curr2next);

            float etaIn, etaOut;
            if (path.bounces[i].mesh->getBSDF()->isConductor()) // conductor reflection: always above the normal
            {
                if (dotIN.vMax <= 0.f || dotON.vMax <= 0.f)
                    return false;
                etaIn = etaOut = 1.f;
            } else // dielectric reflection: on the same side. For TRT
            {
                if ((dotIN.vMax <= 0.0 && dotON.vMin >= 0.0) || (dotIN.vMin >= 0.0 && dotON.vMax <= 0.0))
                    return false;
                etaIn = etaOut = 1.f;
            }

            Interval3D hbox = (ibox + obox).normalized();
            // if (!(hbox - node.norboxRough).coverZero()) {
            //     return false;
            // }
            if (hbox.dot(node.norbox).vMax < node.cosThreshold) {
                return false;
            }
            Interval3D reflectBox = reflect(ibox, node.norboxRough);
            obox = obox.intersect(reflectBox);
            if (obox.isEmpty())
                return false;
            curr2pre = -curr2next;
            ibox = -obox;
        }
        return true;
    }

    int findThickest(const IntervalPath &path, float &thickness) const {
        // find thickest node
        int shouldSplitIdx = -1;
        thickness = 0.f;
        for (size_t i = 0; i < path.bounces.size(); i++) {
            const auto &node = *path.bounces[i].pnode;
            if (!node.tri && node.extent >= thickness) {
                shouldSplitIdx = i;
                thickness = node.extent;
            }
        }
        return shouldSplitIdx;
    }
    void subdivide(std::deque<IntervalPath> &deq, const IntervalPath &path, const int &bouncIdx) const {
        const auto &subNodes = path.bounces[bouncIdx].pnode->children;
        for (int i = 0; i < 4; i++) {
            if (subNodes[i] != nullptr) {
                IntervalPath subPath(path);
                subPath.bounces[bouncIdx].pnode = subNodes[i].get();
                deq.push_back(subPath);
            }
        }
    }
    void subdivide(std::stack<IntervalPath> &stack, const IntervalPath &path, const int &bouncIdx) const {
        const auto &subNodes = path.bounces[bouncIdx].pnode->children;
        for (int i = 0; i < 4; i++) {
            if (subNodes[i] != nullptr) {
                IntervalPath subPath(path);
                subPath.bounces[bouncIdx].pnode = subNodes[i].get();
                stack.push(subPath);
            }
        }
    }

    bool solvePathcut(IntervalPath path, std::vector<Point> &points, std::vector<Vector> &normals) {
        std::vector<std::array<VertInfo, 3>> reflectorTris;
        for (size_t i = 0; i < path.bounces.size() - 1; i++) {
            reflectorTris.emplace_back();
            for (int triVertIdx = 0; triVertIdx < 3; triVertIdx++) {
                const auto &reflTri = path.bounces[i].mesh->getTriangles()[path.bounces[i].pnode->tri->triIdx];
                auto pos = path.bounces[i].mesh->getVertexPositions()[reflTri.idx[triVertIdx]];
                auto nor = path.bounces[i].mesh->getVertexNormals()[reflTri.idx[triVertIdx]];

                reflectorTris.back()[triVertIdx].pos = {pos.x, pos.y, pos.z};
                reflectorTris.back()[triVertIdx].normal = {nor.x, nor.y, nor.z};
            }
        }
        Point recvCenter(0.f);
        for (int triVertIdx = 0; triVertIdx < 3; triVertIdx++) {
            const auto &tri = path.bounces.back().mesh->getTriangles()[path.bounces.back().pnode->tri->triIdx];
            auto pos = path.bounces.back().mesh->getVertexPositions()[tri.idx[triVertIdx]];
            recvCenter += pos;
        }
        recvCenter /= 3;
        const auto &recvNode = path.bounces.back().pnode;
        const auto &lastReflNode = path.bounces[path.bounces.size() - 2].pnode;
        Vector refl2recv = recvCenter - lastReflNode->center;
        float dot1 = dot(refl2recv, lastReflNode->tri->n);
        // offset the receiver point so that it is on the front side of the reflector
        if (dot1 < 0) {
            Vector t = cross(recvNode->tri->n, lastReflNode->tri->n);
            if (t.length() != 0) {
                Vector d = cross(normalize(t), recvNode->tri->n);
                float dot2 = dot(lastReflNode->tri->n, d);
                recvCenter = recvCenter + d * (-std::min(0.f, dot1) / dot2 + recvNode->extent / 10.f);
            }
        }

        // newton solver
        std::vector<VertInfo> reflVerts;
        std::array<float, 3> recvCenterArray = {recvCenter.x, recvCenter.y, recvCenter.z};
        std::array<float, 3> lightPosArray = {path.lightPos.x, path.lightPos.y, path.lightPos.z};
        switch (path.bounces.size()) {
        case 2:
            VertInfo reflVert;
            solveOneBounce(reflVert, reflectorTris[0], lightPosArray, recvCenterArray, m_newton ? 0.001 : 1000000);
            reflVerts.push_back(reflVert);
            break;
        case 3:
            VertInfo reflVert1, reflVert2;
            solveTwoBounce(reflVert1, reflVert2, reflectorTris[0], reflectorTris[1], lightPosArray, recvCenterArray, m_newton ? 0.001 : 100000);
            reflVerts.push_back(reflVert1);
            reflVerts.push_back(reflVert2);
            break;
        default:
            return false;
            break;
        }
        for (size_t i = 0; i < reflVerts.size(); i++) {
            points.emplace_back(reflVerts[i].pos[0], reflVerts[i].pos[1], reflVerts[i].pos[2]);
            normals.emplace_back(reflVerts[i].normal[0], reflVerts[i].normal[1], reflVerts[i].normal[2]);
        }
        points.emplace_back(recvCenter);
        normals.emplace_back(recvNode->tri->n);
        if (m_newton) {
            Vector wi = normalize(path.lightPos - points.front());
            for (size_t i = 0; i < points.size() - 1; i++) {
                Vector wo = normalize(points[i + 1] - points[i]);
                Vector h = normalize(wi + wo);
                float cosine = dot(h, normals[i]);
                if (cosine < path.bounces[i].pnode->cosThreshold) {
                    return false;
                }
                wi = -wo;
            }
        }
        return true;
    }
    bool validateVisibility(const std::vector<Point> &points) const {
        return true;
        for (size_t idx = 0; idx < points.size() - 1; idx++) {
            const auto &curr = points[idx];
            const auto &next = points[idx + 1];

            Vector dir = next - curr;
            auto len = dir.length();
            if (len <= 0.f) {
                return false;
            }

            dir /= len;
            Ray r(curr, dir, 0.01, len - ShadowEpsilon, 0.f);
            if (m_scene->rayIntersect(r)) {
                return false;
            }
        }
        return true;
    }

    void computeSGs(std::vector<SGShape> &sgShapes, const IntervalPath &intervalPath, const std::vector<Point> &points, const std::vector<Vector> &normals) {
        // approximate light transport with SG
        SG gli;
        if (intervalPath.lightType == 0) {
            // point light
            gli = SG::sgLight(0.01f, intervalPath.lightIntensity, intervalPath.lightPos - points[0]);
        } else if (intervalPath.lightType == 1) {
            // area light
            float radius = intervalPath.lightBound.extents().length() / 2;
            gli = SG::sgLight(radius, intervalPath.lightIntensity * (radius * radius), intervalPath.lightPos - points[0]);
        } else if (intervalPath.lightType == 3) {
            // spot
            Vector spotDir = normalize(intervalPath.lightDirBound.centerVector());
            Vector first2light = intervalPath.lightPos - points[0];
            if (dot(spotDir, normalize(-first2light)) < cos(intervalPath.lightCutoff)) {
                return;
            }
            gli = SG::sgLight(0.01f, intervalPath.lightIntensity, first2light);
        }

        Point lightPos = intervalPath.lightPos;
        for (size_t i = 0; i < intervalPath.bounces.size() - 1; i++) {
            const auto &reflPoint = points[i];
            const auto &recvPoint = points[i + 1];
            const auto &reflN = normals[i];
            const auto &reflBounce = intervalPath.bounces[i];
            const auto &recvBounce = intervalPath.bounces[i + 1];
            const auto &reflMesh = reflBounce.mesh;
            const auto &recvMesh = recvBounce.mesh;
            const auto &reflPNode = reflBounce.pnode;
            const auto &recvPNode = recvBounce.pnode;
            int reflTriIdx = reflBounce.pnode->tri->triIdx;
            int recvTriIdx = recvBounce.pnode->tri->triIdx;
            float reflFocalU = reflBounce.pnode->tri->focalLengthU;
            float reflFocalV = reflBounce.pnode->tri->focalLengthV;

            Vector wi = lightPos - reflPoint;
            float lightDist = wi.length();
            wi /= lightDist;
            Vector wo = normalize(recvPoint - reflPoint);
            Vector h = normalize(wi + wo);
            // curved mirror approximation
            const auto &tangent = reflMesh->getUVTangents()[reflTriIdx];
            Vector reflS = normalize(cross(normalize(tangent.dpdv), h));
            Vector reflT = normalize(cross(h, reflS));
            Frame tangentFrame(reflS, reflT, h);
            float d_o, hoU, hoV;
            ParallaxStruct parallaxInfo = m_parallax ? ParallaxStruct(reflPoint, lightPos, tangentFrame, reflFocalU, reflFocalV, d_o, hoU, hoV) : ParallaxStruct();

            // shading point too close to image, consider reflector as plane
            float imageDistU = INFINITY, imageDistV = INFINITY;
            parallaxInfo.imageDist(recvPoint, imageDistU, imageDistV);
            if (imageDistU < 0.5f) {
                parallaxInfo.diU *= 500;
                parallaxInfo.hiU *= 500;
            }
            if (imageDistV < 0.5f) {
                parallaxInfo.diV *= 500;
                parallaxInfo.hiV *= 500;
            }

            SG reflBrdfSG = SG::brdfSlice(reflN, wo, reflPNode->maxRoughness, reflMesh->getBSDF());
            gli = SG::convolveApproximation(gli, reflBrdfSG, reflN);
            gli.p = -wo;
            // if (!(gli.c.isValid() && std::isfinite(gli.lambda) && gli.lambda > 0) ||
            //     gli.c.isZero()) {
            //     return;
            // }
            // if (recvBounce.guided) {
            //     auto &sgMixtures = sgShapes[recvMesh->getID()[0]].sgMixtures;
            //     const auto &parent = recvPNode->parent;
            //     if (parent) {
            //         omp_set_lock(&m_lock);
            //         putSG(sgMixtures, parent, {gli, parallaxInfo, 1});
            //         omp_unset_lock(&m_lock);
            //     } else {
            //         omp_set_lock(&m_lock);
            //         m_sgCount++;
            //         sgMixtures[recvTriIdx].sgLights.push_back(
            //             {gli, parallaxInfo, 1});
            //         omp_unset_lock(&m_lock);
            //     }
            // }
            // gli.c *= 10;
            // SG region = SG::region(recvPoint, reflPoint, reflN, reflPNode->extent * reflPNode->extent / 2);
            // SG product = SG::product(gli, region);
            // gli.lambda = std::min(product.lambda, 50000.f);
            // gli.c = product.c;
            if (!(gli.c.isValid() && std::isfinite(gli.lambda) && gli.lambda > 0) ||
                gli.c.isZero()) {
                return;
            }
            if (recvBounce.guided) {
                auto &sgMixtures = sgShapes[recvMesh->getID()[0]].sgMixtures;
                const auto &parent = recvPNode->parent;
                if (parent) {
                    omp_set_lock(&m_lock);
                    putSG(sgMixtures, parent, {gli, parallaxInfo});
                    omp_unset_lock(&m_lock);
                } else {
                    omp_set_lock(&m_lock);
                    // add a smooth lobe into the GMM
                    if (sgMixtures[recvTriIdx].sgLights.empty()) {
                        sgMixtures[recvTriIdx].sgLights.push_back({SG(recvPNode->norbox.centerVector(), 2, Spectrum(0.f)), ParallaxStruct()});
                        m_sgCount++;
                    }
                    m_sgCount++;
                    sgMixtures[recvTriIdx].sgLights.push_back(
                        {gli, parallaxInfo});
                    omp_unset_lock(&m_lock);
                }
            }
            if (i < intervalPath.bounces.size() - 2) {
                // isotropic curved mirror approximation
                float focalAvg = (std::isinf(reflFocalU) || std::isinf(reflFocalV)) ? INFINITY : (reflFocalU + reflFocalV) / 2.f;
                float nDotL = dot(reflN, wi);
                float d_o1 = nDotL * lightDist,
                      d_i1 = 1 / (1 / focalAvg - 1 / d_o1);
                lightDist = d_i1 / nDotL;
                lightPos = reflPoint + lightDist * wo;
            }
        }

        // gli.lambda /= 1 / r + 1;
        // gli.c /= 1 / r + 1;
        // gli.lambda *= min(imageDistU, imageDistV);
        // gli.c *= r * r;
    }
    void putSG(std::vector<SGMixture> &sgMixtures, PNode *node, const SGLight &sg) {
        if (node->tri) {
            // add a smooth lobe into the GMM
            if (sgMixtures[node->tri->triIdx].sgLights.empty()) {
                sgMixtures[node->tri->triIdx].sgLights.push_back({SG(node->norbox.centerVector(), 2, Spectrum(0.f)), ParallaxStruct()});
                m_sgCount++;
            }
            m_sgCount++;
            sgMixtures[node->tri->triIdx].sgLights.push_back(sg);
            return;
        }
        for (const auto &child : node->children) {
            if (child) {
                putSG(sgMixtures, child.get(), sg);
            }
        }
    }
    // static Interval3D computeHalfV(const Interval3D &in, const Interval3D &out,
    //                                int type, float etaIn, float etaOut) {
    //     switch (type) {
    //     case -1:
    //         return (in + out).normalized();
    //     case -2:
    //         return -(in + out).normalized();
    //     case 1:
    //         return -(in * etaIn + out * etaOut).normalized();
    //     }
    // }

    // static Interval3D refract(const Interval3D &wi, const Interval3D &n, float eta) {
    //     Interval1D cosThetaI = wi.dot(n);
    //     if (cosThetaI.vMin > 0)
    //         eta = 1.0f / eta;

    //     if (cosThetaI.vMax > 0) {
    //         return Interval3D(Point(-1e10), Point(1e10));
    //     }

    //     Interval1D cosThetaTSqr = Interval1D(1.0f) - (Interval1D(1.0f) - cosThetaI * cosThetaI) * (eta * eta);
    //     return n * (cosThetaI * eta - cosThetaTSqr.sqrt() * (signbit(cosThetaI.vMin) ? -1.f : 1.f)) - wi * eta;
    // }
};