#include "pathcut.h"
#ifdef MYDEBUG
#pragma GCC optimize("O0")
#endif

/* Wrapper class of the precomputation functions */
class PathcutWrapper {
    std::vector<GuidedShape> m_sgShapes;
    float m_mergeAngleThreshold;
    float m_mergeDistThreshold;

    float m_bsdfSamplingFraction;
    float m_sgCutoff;

    bool m_parallax;
    bool m_sds;
    bool m_newton;

    int m_maxDepth;

public:
    bool guideSDS() const {
        return m_sds;
    }
    void initGuidingParams(const Properties &props) {
        m_mergeAngleThreshold = props.getFloat("mergeAngleThreshold", 0.f);
        m_mergeDistThreshold = props.getFloat("mergeDistThreshold", 0.f);
        m_bsdfSamplingFraction = props.getFloat("samplingFraction", 0.5f);
        m_sgCutoff = props.getFloat("sgCutoff", 0.01f);
        m_parallax = props.getBoolean("parallax", true);
        m_sds = props.getBoolean("SDS", false);
        m_newton = props.getBoolean("newton", true);
        m_maxDepth = props.getInteger("causticBounce", 3);
    }
    void precomputePathCuts(const Scene *scene, const Point &camPos, const Vector &camDir) {
        // Build scene hierarchy for path cut pruning.
        std::vector<PTree> ptrees;
        const auto &meshes = scene->getMeshes();
        int receiverID = 0;
        for (const auto &mesh : meshes) {
            if (mesh->getProperties().getBoolean("guided", false) || mesh->getProperties().getBoolean("receiver", false)) {
                ptrees.emplace_back(mesh, m_sgCutoff);
                m_sgShapes.emplace_back();
                m_sgShapes.back().sgMixtures.resize(mesh->getPrimitiveCount());
                mesh->setID(std::string(1, char(receiverID)));
                receiverID++;
            } else if (mesh->getProperties().getBoolean("reflector", false) || mesh->getProperties().getBoolean("refractor", false)) {
                ptrees.emplace_back(mesh, m_sgCutoff);
                mesh->setID(std::string(1, char(-1)));
            } else {
                mesh->setID(std::string(1, char(-1)));
            }
        }
        std::cout << "guided shape num: " << receiverID << "\n";

        //Precomputation:
        //  Finding leaf path cuts.
        //  Folving for representative paths.
        //  Approximating incident radiance distribution with Spherical Gaussians.
        Precomputation(m_sgShapes, scene, ptrees, camPos, camDir, 2, m_mergeAngleThreshold, m_mergeDistThreshold, m_sds, m_parallax, m_newton);
        for (int i = 3; i < m_maxDepth; i++) {
            //for multiple bounces
            Precomputation(m_sgShapes, scene, ptrees, camPos, camDir, i, m_mergeAngleThreshold, m_mergeDistThreshold, false, m_parallax, m_newton);
        }
        for (size_t i = 0; i < m_sgShapes.size(); i++) {

#ifndef MYDEBUG
#pragma omp parallel for schedule(dynamic)
#endif
            //process the GMMs for sampling.
            for (size_t j = 0; j < m_sgShapes[i].sgMixtures.size(); j++) {
                m_sgShapes[i].sgMixtures[j].normalizeForSampling();
                m_sgShapes[i].sgMixtures[j].computeWeights();
            }
        }
        int mergedSGCount = 0;
        for (size_t i = 0; i < m_sgShapes.size(); i++) {
            for (size_t j = 0; j < m_sgShapes[i].sgMixtures.size(); j++) {
                mergedSGCount += m_sgShapes[i].sgMixtures[j].sgLights.size();
            }
        }
        size_t memory = sizeof(SGLight) * mergedSGCount;
        std::cout << "total SG count for rendering: " << mergedSGCount << "\n";
        std::cout << "memory: " << memory / 1024 << " KB.\n";
    }
    Spectrum sampleMat(Float &woPdf, const BSDF *bsdf, BSDFSamplingRecord &bRec, mitsuba::ETransportMode mode, Point2 sample) const {
        Spectrum result;
        Float bsdfPdf = 0, sTreePdf = 0;

        auto shapeIdx = bRec.its.shape->getID()[0];
        auto triIdx = bRec.its.primIndex;
        if (m_bsdfSamplingFraction == 1.f ||
            shapeIdx == char(-1) ||
            bRec.its.shape->isEmitter() ||
            mode != ERadiance ||
            m_sgShapes[shapeIdx].sgMixtures[triIdx].sgLights.size() == 0) {
            result = bsdf->sample(bRec, woPdf, sample);
            return result;
        } else {
            const auto &sgMixture = m_sgShapes[shapeIdx].sgMixtures[triIdx];
            if (sample.x < m_bsdfSamplingFraction) {
                sample.x /= m_bsdfSamplingFraction;
                result = bsdf->sample(bRec, bsdfPdf, sample);
                if (result.isZero()) {
                    woPdf = 0;
                    return Spectrum{0.0f};
                }
                result *= bsdfPdf;
                Vector wo = bRec.its.toWorld(bRec.wo);
                sTreePdf = sgMixture.pdfWeighted(bRec.its.p, wo);
            } else {
                sample.x = (sample.x - m_bsdfSamplingFraction) / (1 - m_bsdfSamplingFraction);
                Vector wo = sgMixture.sampleWeighted(bRec.its.p, sample);
                bRec.eta = 1.0f;
                bRec.sampledComponent = 0;
                bRec.sampledType = BSDF::EGlossyReflection;

                sTreePdf = sgMixture.pdfWeighted(bRec.its.p, wo);
                bRec.wo = bRec.its.toLocal(wo);
                result = bsdf->eval(bRec);
                bsdfPdf = bsdf->pdf(bRec);
            }
        }
        woPdf = m_bsdfSamplingFraction * bsdfPdf + (1 - m_bsdfSamplingFraction) * sTreePdf;
        if (result.isZero() || woPdf == 0) {
            return Spectrum{0.0f};
        }

        return result / woPdf;
    }

    Float pdfMat(const BSDF *bsdf, BSDFSamplingRecord &bRec, mitsuba::ETransportMode mode, EMeasure measure = ESolidAngle) const {
        auto shapeIdx = bRec.its.shape->getID()[0];
        auto triIdx = bRec.its.primIndex;
        if (m_bsdfSamplingFraction == 1.f ||
            shapeIdx == char(-1) ||
            bRec.its.shape->isEmitter() ||
            mode != ERadiance ||
            m_sgShapes[shapeIdx].sgMixtures[triIdx].sgLights.size() == 0) {
            return bsdf->pdf(bRec, measure);
        } else {
            const auto &sgMixture = m_sgShapes[shapeIdx].sgMixtures[triIdx];
            float sTreePdf = sgMixture.pdfWeighted(bRec.its.p, bRec.its.toWorld(bRec.wo));
            Float bsdfPdf = bsdf->pdf(bRec);
            Float woPdf = m_bsdfSamplingFraction * bsdfPdf + (1 - m_bsdfSamplingFraction) * sTreePdf;
            return woPdf;
        }
    }

    Spectrum evalSGs(const Point &pos, const Intersection &its, const Vector &dir) const {
        const auto &sgMixtures = m_sgShapes[its.shape->getID()[0]].sgMixtures[its.primIndex];
        return sgMixtures.evalParallax(pos, dir);
    }
};