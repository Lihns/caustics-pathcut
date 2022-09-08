#pragma once
#include <mitsuba/core/aabb.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/trimesh.h>

#include "interval.h"

using namespace mitsuba;

void Barycentric(Point p, Point a, Point b, Point c, float &u, float &v, float &w) {
    Vector v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}
struct PTriNode {
    Vector n;
    float focalLengthU, focalLengthV;
    int triIdx;
};
struct PNode {
    Interval3D posBox;
    Interval3D norbox;
    Interval3D norboxRough;
    float cosThreshold;
    Point center;
    float extent;
    float minRoughness;
    float maxRoughness;
    std::unique_ptr<PTriNode> tri;
    PNode *parent;

    std::array<std::unique_ptr<PNode>, 4> children;
};

class PTree {
public:
    std::unique_ptr<PNode> shapeRoot;
    std::vector<PNode *> m_leafNodes;
    const TriMesh *trimesh;

public:
    PTree() {}
    PTree(const TriMesh *mesh, float sgCutOff) {
        ref<Timer> timer = new Timer();
        
        trimesh = mesh;
        std::vector<uint32_t> triNodeIndices;
        AABB shapeAABB;
        shapeAABB.expandBy(mesh->getAABB());
        for (size_t triIdx = 0; triIdx < mesh->getTriangleCount(); triIdx++) {
            m_leafNodes.push_back(buildTriangleNode(mesh, triIdx, sgCutOff));
            triNodeIndices.push_back(m_leafNodes.size() - 1);
        }
        shapeRoot = std::unique_ptr<PNode>(buildTreeForMeshMedian(shapeAABB.getExtents()[shapeAABB.getLargestAxis()],
                                                                  shapeAABB,
                                                                  triNodeIndices, 0, triNodeIndices.size(), nullptr));

        std::cout << "\nptree done in " << timer->getSeconds() << "s\n";
    }
    PTree(const TriMesh *mesh, float sgCutOff, int triIdx) {
        ref<Timer> timer = new Timer();

        std::vector<uint32_t> triNodeIndices;
        AABB shapeAABB;
        m_leafNodes.push_back(buildTriangleNode(mesh, triIdx, sgCutOff));
        shapeAABB = mesh->getTriangles()[triIdx].getAABB(mesh->getVertexPositions());
        triNodeIndices.push_back(m_leafNodes.size() - 1);

        shapeRoot = std::unique_ptr<PNode>(buildTreeForMeshMedian(shapeAABB.getExtents()[shapeAABB.getLargestAxis()],
                                                                  shapeAABB,
                                                                  triNodeIndices, 0, triNodeIndices.size(), nullptr));

        std::cout << "\nptree done in " << timer->getSeconds() << "s\n";
    }

private:
    void focalUV(const Frame &tangentFrame, float posExtentU, float posExtentV,
                 const Point &a, const Point &b, const Point &c,
                 const Vector &na, const Vector &nb, const Vector &nc,
                 float &focalU, float &focalV) {
        Point center = (a + b + c) / 3.f;
        Point positiveP = center + posExtentU / 2 * tangentFrame.s;
        Point negativeP = center - posExtentU / 2 * tangentFrame.s;
        float u, v, w;
        Barycentric(positiveP, a, b, c, u, v, w);
        Vector np = normalize(u * na + v * nb + w * nc);
        Barycentric(negativeP, a, b, c, u, v, w);
        Vector nn = normalize(u * na + v * nb + w * nc);
        float nExtentU = tangentFrame.toLocal(np - nn).x;
        focalU = -posExtentU * (1.f / (2.f * nExtentU) - nExtentU / 8.f);
        positiveP = center + posExtentV / 2 * tangentFrame.t;
        negativeP = center - posExtentV / 2 * tangentFrame.t;
        Barycentric(positiveP, a, b, c, u, v, w);
        np = normalize(u * na + v * nb + w * nc);
        Barycentric(negativeP, a, b, c, u, v, w);
        nn = normalize(u * na + v * nb + w * nc);
        float nExtentV = tangentFrame.toLocal(np - nn).y;
        focalV = -posExtentV * (1.f / (2.f * nExtentV) - nExtentV / 8.f);
    }
    PNode *buildTriangleNode(const TriMesh *mesh, int triIdx, float sgCutOff) {
        const auto &bsdf = mesh->getBSDF();
        if (!bsdf->hasComponent(BSDF::ESmooth) || bsdf->hasComponent(BSDF::ETransmission)) {
            std::cout << "bsdf not supported!\n";
            return nullptr;
        }
        const Point *vertexPositions = mesh->getVertexPositions();
        const Normal *vertexNormals = mesh->getVertexNormals();
        const TangentSpace *tangents = mesh->getUVTangents();
        const Triangle *triangles = mesh->getTriangles();
        const auto &triangle = triangles[triIdx];

        float roughness = bsdf->hasComponent(BSDF::EGlossy) ? bsdf->getProperties().getFloat("alpha") : 1;
        const float m2 = roughness * roughness;
        const float logPiM2 = std::log(sgCutOff);
        float cosThres = std::max(1e-10f, logPiM2 * m2 / 2 + 1);
        float angle = radToDeg(acos(cosThres));

        Point center = (vertexPositions[triangle.idx[0]] + vertexPositions[triangle.idx[1]] + vertexPositions[triangle.idx[2]]) / 3;
        Vector side1 = vertexPositions[triangle.idx[0]] - vertexPositions[triangle.idx[1]];
        Vector side2 = vertexPositions[triangle.idx[0]] - vertexPositions[triangle.idx[2]];
        Vector triN = normalize(cross(side1, side2));
        const auto &tan = tangents[triIdx];
        Vector s, t;
        s = normalize(tan.dpdu);
        t = normalize(cross(triN, s));
        Frame tangentFrame(s, t, triN);

        Interval1D posBoundU, posBoundV;

        Interval3D norbox;
        Interval3D norboxRough;
        auto posaabb = triangle.getAABB(vertexPositions);
        Interval3D posbox(posaabb.min, posaabb.max);

        for (int i = 0; i < 3; i++) {
            const Vector &n = vertexNormals[triangle.idx[i]];
            Vector pLocal = tangentFrame.toLocal(vertexPositions[triangle.idx[i]] - center);
            // Vector nLocal = tangentFrame.toLocal(n);
            posBoundU.expand(pLocal.x);
            posBoundV.expand(pLocal.y);

            norbox.expand(n);
            norboxRough.expand(n);
            if (mesh->hasUVTangents()) {
                norboxRough.expand(Transform::rotate(s, angle)(n));
                norboxRough.expand(Transform::rotate(s, -angle)(n));
                norboxRough.expand(Transform::rotate(t, angle)(n));
                norboxRough.expand(Transform::rotate(t, -angle)(n));
            } else {
                norboxRough.expand(Transform::rotate(Vector(1, 0, 0), angle)(n));
                norboxRough.expand(Transform::rotate(Vector(1, 0, 0), -angle)(n));
                norboxRough.expand(Transform::rotate(Vector(0, 1, 0), angle)(n));
                norboxRough.expand(Transform::rotate(Vector(0, 1, 0), -angle)(n));
                norboxRough.expand(Transform::rotate(Vector(0, 0, 1), angle)(n));
                norboxRough.expand(Transform::rotate(Vector(0, 0, 1), -angle)(n));
            }
        }
        float posExtentU = posBoundU.length(),
              posExtentV = posBoundV.length();
        //   nExtentU = nBoundU.length(),
        //   nExtentV = nBoundV.length();
        float focalLengthU, focalLengthV;
        focalUV(tangentFrame, posExtentU, posExtentV,
                vertexPositions[triangle.idx[0]], vertexPositions[triangle.idx[1]], vertexPositions[triangle.idx[2]],
                vertexNormals[triangle.idx[0]], vertexNormals[triangle.idx[1]], vertexNormals[triangle.idx[2]],
                focalLengthU, focalLengthV);
        PTriNode *tri = new PTriNode{
            triN,
            focalLengthU,
            focalLengthV,
            triIdx,
        };
        return new PNode{
            posbox,
            norbox.normalized(),
            norboxRough,
            cosThres,
            center,
            posaabb.getExtents().length(),
            roughness,
            roughness,
            std::unique_ptr<PTriNode>(tri),
            nullptr,
            {nullptr},
        };
    }

    PNode *buildTreeForMeshMedian(const float &largestExtent, const AABB &aabb, std::vector<uint32_t> &triNodeIndices, size_t beginIdx, size_t endIdx, PNode *parent) {
        if (endIdx - beginIdx == 1) {
            m_leafNodes[triNodeIndices[beginIdx]]->parent = parent;
            return m_leafNodes[triNodeIndices[beginIdx]];
        }

        PNode *node = new PNode();
        if (endIdx - beginIdx <= 4) {
            for (size_t i = 0; i < endIdx - beginIdx; i++) {
                // construct a node for a triangle
                node->children[i] = std::unique_ptr<PNode>(m_leafNodes[triNodeIndices[i + beginIdx]]);
                node->children[i]->parent = node;
            }
        } else {
            std::array<AABB, 2> temp = {};
            size_t median = splitMidanPos(aabb, temp[0], temp[1], triNodeIndices, beginIdx, endIdx);

            std::array<AABB, 4> childAABB = {};
            size_t leftMedian = splitMidanPos(temp[0], childAABB[0], childAABB[1], triNodeIndices, beginIdx, median);
            size_t rightMedian = splitMidanPos(temp[1], childAABB[2], childAABB[3], triNodeIndices, median, endIdx);

            node->children[0] = std::unique_ptr<PNode>(buildTreeForMeshMedian(largestExtent, childAABB[0], triNodeIndices, beginIdx, leftMedian, node));
            node->children[1] = std::unique_ptr<PNode>(buildTreeForMeshMedian(largestExtent, childAABB[1], triNodeIndices, leftMedian, median, node));
            node->children[2] = std::unique_ptr<PNode>(buildTreeForMeshMedian(largestExtent, childAABB[2], triNodeIndices, median, rightMedian, node));
            node->children[3] = std::unique_ptr<PNode>(buildTreeForMeshMedian(largestExtent, childAABB[3], triNodeIndices, rightMedian, endIdx, node));
        }
        node->minRoughness = 1.f;
        node->maxRoughness = 0.f;
        node->cosThreshold = 1.f;
        int nChild = 0;
        for (size_t i = 0; i < node->children.size(); i++) {
            const auto &child = node->children[i];
            if (child != nullptr) {
                nChild++;
                node->posBox = minmax(node->posBox, child->posBox);
                node->norbox = minmax(node->norbox, child->norbox);
                node->norboxRough = minmax(node->norboxRough, child->norboxRough);
                node->cosThreshold = std::min(node->cosThreshold, child->cosThreshold);
                node->minRoughness = std::min(child->minRoughness, node->minRoughness);
                node->maxRoughness = std::max(child->maxRoughness, node->maxRoughness);
            }
        }
        node->center = node->posBox.center();
        node->extent = node->posBox.extents().length();
        node->parent = parent;
        node->tri = nullptr;
        return node;
    }

    size_t splitMidanPos(const AABB &aabb,
                         AABB &leftAABB, AABB &rightAABB,
                         std::vector<uint32_t> &triNodeIndices, size_t beginIdx, size_t endIdx) {
        int axis = aabb.getLargestAxis();

        auto centerComparator = [&](uint32_t idx1, uint32_t idx2) {
            return m_leafNodes[idx1]->center[axis] < m_leafNodes[idx2]->center[axis];
        };
        // from small to large
        std::sort(triNodeIndices.begin() + beginIdx, triNodeIndices.begin() + endIdx, centerComparator);
        size_t median = (beginIdx + endIdx) / 2;
        for (size_t i = beginIdx; i < median; i++) {
            const Point &center = m_leafNodes[triNodeIndices[i]]->center;
            leftAABB.expandBy(center);
        }
        for (size_t i = median; i < endIdx; i++) {
            const Point &center = m_leafNodes[triNodeIndices[i]]->center;
            rightAABB.expandBy(center);
        }
        return median;
    }
};