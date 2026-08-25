#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tidegbm {

class TideError : public std::runtime_error {
public:
    explicit TideError(const std::string& m) : std::runtime_error(m) {}
};

using f64 = double;
using i32 = int32_t;
using i64 = int64_t;
using u16 = uint16_t;
using u8 = uint8_t;

constexpr u16 kMissingBin = 0;
constexpr int kMaxBinCap = 256;
constexpr uint32_t kFormatVersion = 3;
constexpr char kMagic[4] = {'T', 'G', 'B', 'M'};

enum class SplitType : int8_t { Axis = 0, Pair = 1, Cat = 2, Regime = 3 };
enum class ObjectiveKind : int8_t {
    MSE = 0,
    Huber = 1,
    Quantile = 2,
    Logloss = 3,
    RankListNet = 4,
    RankPairwise = 5,
    Multiclass = 6,
    Custom = 7
};

}  // namespace tidegbm
