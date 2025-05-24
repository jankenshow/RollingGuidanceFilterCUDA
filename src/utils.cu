#include "utils.h"

namespace rgf {

__host__ __device__ int clamp(int v, int low, int high) {
  return max(low, min(v, high));
}

}  // namespace rgf