#ifndef PROJCORECUDACORES_CU_H_85E8271D
#define PROJCORECUDACORES_CU_H_85E8271D

#include "Constants.h"
#include <cuda_runtime.h>

typedef struct matrix { REAL m[2][2]; } matrix;

class Mat2Mult {
    public:
        static __device__ inline matrix identity () {
            matrix id = { {{1.0, 0.0}, {0.0, 1.0}} };
            return id;
        }

        static __device__ inline matrix apply(const matrix t1, const matrix t2) {
            matrix prod = { {
                { t1.m[0][0] * t2.m[0][0] + t1.m[0][1] * t2.m[1][0],
                  t1.m[0][0] * t2.m[0][1] + t1.m[0][1] * t2.m[1][1] },
                { t1.m[1][0] * t2.m[0][0] + t1.m[1][1] * t2.m[1][0],
                  t1.m[1][0] * t2.m[0][1] + t1.m[1][1] * t2.m[1][1] }
            } };

            return prod;
        }
};

#endif

