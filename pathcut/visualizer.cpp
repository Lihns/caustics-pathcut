#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>

#include "pathcut_wrapper.h"

MTS_NAMESPACE_BEGIN

class PathcutVisualizer : public MonteCarloIntegrator {
private:
    PathcutWrapper pwrapper;

    Transform m_camTrafo;
    Point m_visualizerPos;
    Intersection m_its;

public:
    PathcutVisualizer(const Properties &props) : MonteCarloIntegrator(props) {
        pwrapper.initGuidingParams(props);
        m_camTrafo = props.getTransform("camTrafo");
    }

    PathcutVisualizer(Stream *stream, InstanceManager *manager) : MonteCarloIntegrator(stream, manager) {}

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();
        int integratorResID = sched->registerResource(this);

        m_visualizerPos = sensor->getWorldTransform()->eval(0.f)(Point(0.f));

        Point camPos = m_camTrafo(Point(0.f));
        Vector camDir = m_camTrafo(Vector(0.f, 0.f, 1.f));

        const auto &meshes = scene->getMeshes();
        for (size_t i = 0; i < meshes.size(); i++) {
            meshes[i]->setID(std::string(1, char(i)));
        }

        Ray r(camPos, normalize(m_visualizerPos - camPos), (m_visualizerPos - camPos).length() - 0.1, (m_visualizerPos - camPos).length() + 0.1, 0.f);
        Intersection its;
        scene->rayIntersect(r, its);
        if (its.isValid() && (its.shape->getProperties().getBoolean("guided", false) || its.shape->getProperties().getBoolean("receiver", false))) {
            m_its = its;
        } else {
            return false;
        }

        pwrapper.precomputePathCuts(scene, camPos, camDir);

        ref<ParallelProcess> proc;
        proc = new BlockedRenderProcess(job, queue, scene->getBlockSize());
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

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        Spectrum Li = pwrapper.evalSGs(m_visualizerPos, m_its, r.d);
        return Li;
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "PathcutVisualizer[" << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(PathcutVisualizer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathcutVisualizer, "Pathcut visualizer");
MTS_NAMESPACE_END