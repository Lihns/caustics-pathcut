#include <array>
#include <vector>

struct VertInfo {
    std::array<float, 3> pos;
    std::array<float, 3> normal;
};

float solveOneBounce(VertInfo &reflVert, const std::array<VertInfo, 3> &triInfo,
                     const std::array<float, 3> &lightPos, const std::array<float, 3> &camPos, float errorThreshold);

float solveTwoBounce(VertInfo &reflVert1, VertInfo &reflVert2, const std::array<VertInfo, 3> &triInfo1, const std::array<VertInfo, 3> &triInfo2,
                     const std::array<float, 3> &lightPos_, const std::array<float, 3> &camPos_, float errorThreshold);