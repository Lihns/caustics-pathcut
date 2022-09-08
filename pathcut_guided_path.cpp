/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob
    Copyright (c) 2017 by ETH Zurich, Thomas Mueller.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>

#include "pathcut/pathcut_wrapper.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

class PathcutGuidedPathTracer : public MonteCarloIntegrator {
private:
    // std::vector<std::vector<STree>> m_sTrees;
    PathcutWrapper pwrapper;

public:
    PathcutGuidedPathTracer(const Properties &props) : MonteCarloIntegrator(props) {
        pwrapper.initGuidingParams(props);
    }

    /// Unserialize from a binary data stream
    PathcutGuidedPathTracer(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) {}

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
                int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        size_t nCores = sched->getCoreCount();
        const Sampler *sampler = static_cast<const Sampler *>(sched->getResource(samplerResID, 0));
        size_t sampleCount = sampler->getSampleCount();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
            nCores == 1 ? "core" : "cores");

        auto trafo = sensor->getWorldTransform()->eval(0.f);
        Point camPos = trafo(Point(0.f));
        Vector camDir = trafo(Vector(0.f, 0.f, 1.f));

        pwrapper.precomputePathCuts(scene, camPos, camDir);

        ref<ParallelProcess> proc = new BlockedRenderProcess(job,
                                                             queue, scene->getBlockSize());
        int integratorResID = sched->registerResource(this);
        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);
        scene->bindUsedResources(proc);
        bindUsedResources(proc);
        sched->schedule(proc);

        m_process = proc;
        sched->wait(proc);
        m_process = NULL;
        sched->unregisterResource(integratorResID);

        return proc->getReturnStatus() == ParallelProcess::ESuccess;
    }

    void renderBlock(const Scene *scene,
                     const Sensor *sensor, Sampler *sampler, ImageBlock *block,
                     const bool &stop, const std::vector<TPoint2<uint8_t>> &points) const {

        size_t sampleCount = sampler->getSampleCount();
        Float diffScaleFactor = 1.0f / std::sqrt((Float)sampleCount);

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
            queryType &= ~RadianceQueryRecord::EOpacity;

        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            if (stop)
                break;

            sampler->generate(offset);

            for (size_t j = 0; j < sampleCount; j++) {
                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                sensorRay.scaleDifferential(diffScaleFactor);

                spec *= Li(sensorRay, rRec);
                block->put(samplePos, spec, rRec.alpha);
                sampler->advance();
            }
        }
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF();

            // if (sNodes.empty() && rRec.depth == 1) {
            //     float data[3] = {0, 1, 0};
            //     return Spectrum(data);
            // }

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            // /* Include radiance from a subsurface scattering model if requested */
            // if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
            //     Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0) || (m_strictNormals && dot(ray.d, its.geoFrame.n) * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling (NEE)               */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if ((rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                            ? (its.shape->getID()[0] == char(-1) ? bsdf->pdf(bRec) : pwrapper.pdfMat(bsdf, bRec, ERadiance))
                                            : 0;

                        /* Weight using the power heuristic */
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
#ifdef SINGLE_BOUNCE
                        if (rRec.depth == m_maxDepth - 1)
#endif
                            Li += throughput * value * bsdfVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */
            Spectrum bsdfWeight;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Float bsdfPdf;
            bool rayHit;

            bsdfWeight = pwrapper.sampleMat(bsdfPdf, bsdf, bRec, ERadiance, bRec.sampler->next2D());

            if (bsdfWeight.isZero())
                break;

            scattered |= bRec.sampledType != BSDF::ENull;

            // /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;

            ray = Ray(its.p, wo, ray.time);
            rayHit = scene->rayIntersect(ray, its);
            if (rayHit) {
                /* Intersected something - check if it was a luminaire */
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = scene->getEnvironmentEmitter();

                if (env) {
                    if (m_hideEmitters && !scattered)
                        break;

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;
                    hitEmitter = true;
                } else {
                    break;
                }
            }
            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = !(bRec.sampledType & BSDF::EDelta) ? scene->pdfEmitterDirect(dRec) : 0;
#ifdef SINGLE_BOUNCE
                if (rRec.depth == m_maxDepth - 1)
#endif
                    Li += throughput * value * miWeight(bsdfPdf, lumPdf);
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float)0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        return Li;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "PathcutGuidedPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(PathcutGuidedPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathcutGuidedPathTracer, "Path tracer guided with pathcut");
MTS_NAMESPACE_END