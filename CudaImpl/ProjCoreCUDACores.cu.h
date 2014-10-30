#ifndef PROJCORECUDACORES_CU_H_85E8271D
#define PROJCORECUDACORES_CU_H_85E8271D

#define IDX2(DIMX, DIMY, X, Y)           ((X)*(DIMY) + (Y))
#define IDX3(DIMX, DIMY, DIMZ, X, Y, Z)  ((X)*(DIMY)*(DIMZ) + (Y)*(DIMZ) + (Z))

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

__device__
void tridag(
    const REAL   *a,   // size [n]
    const REAL   *b,   // size [n]
    const REAL   *c,   // size [n]
    const REAL   *r,   // size [n]
    const int             n,
          REAL   *u,   // size [n]
          REAL   *uu   // size [n] temporary
) {
    int    i, offset;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
    }
#else
    // Hint: X) can be written smth like (once you make a non-constant)
    for(i=0; i<n; i++) a[i] =  u[n-1-i];
    a[0] = a[0] / uu[n-1];
    for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
    for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
}

__global__
void updateParams_large_kernel(
        const unsigned g,
        const REAL alpha,
        const REAL beta,
        const REAL nu,
        unsigned int max_outer,
        unsigned int numX,
        unsigned int numY,
        unsigned int numT,
        REAL *myX,
        REAL *myY,
        REAL *myVarX,
        REAL *myVarY,
        REAL *myTimeline
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= max_outer || tid_x >= numX)
        return;

    for (unsigned j = 0; j < numY; ++j) {
        REAL x        = myX[tid_x],
             y        = myY[j],
             timeline = myTimeline[g];

        myVarX[IDX3(max_outer, numX, numY, tid_outer, tid_x, j)]
            = exp(2.0 * (beta * log(x) + y - 0.5*nu*nu*timeline));

        myVarY[IDX3(max_outer, numX, numY, tid_outer, tid_x, j)]
            = exp(2.0 * (alpha * log(x) + y - 0.5*nu*nu*timeline));

        if (tid_outer == 20 && tid_x == 52 && j == 253) {
            printf("cuda: %.10f %.10f %.10f\n", x, y, timeline);
            printf("cuda: %.10f %.10f\n", log(x), myVarY[IDX3(max_outer, numX, numY, tid_outer, tid_x, j)]);
        }
    }
}

__global__
void initGrid_kernel(
        const REAL s0,
        const REAL logAlpha,
        const REAL dx,
        const REAL dy,
        const REAL myXindex,
        const REAL myYindex,
        const REAL t,
        const unsigned numX,
        const unsigned numY,
        const unsigned numT,
        REAL *myTimeline,
        REAL *myX,
        REAL *myY
) {
    unsigned int tid_i = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid_i < numT) {
        myTimeline[tid_i] = t*tid_i/(numT-1);
    }

    if (tid_i < numX) {
        myX[tid_i] = tid_i*dx - myXindex*dx + s0;
    }

    if (tid_i < numY) {
        myY[tid_i] = tid_i*dy - myYindex*dy + logAlpha;
    }
}

__global__
void initOperator_kernel(REAL *x, REAL *Dxx, const unsigned numX) {
    unsigned int tid_x = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid_x >= numX)
        return;

    if (tid_x == 0 || tid_x == numX-1) {
        Dxx[IDX2(numX,4, tid_x,0)] =  0.0;
        Dxx[IDX2(numX,4, tid_x,1)] =  0.0;
        Dxx[IDX2(numX,4, tid_x,2)] =  0.0;
        Dxx[IDX2(numX,4, tid_x,3)] =  0.0;
    } else {
        REAL dxl = x[tid_x] - x[tid_x-1];
        REAL dxu = x[tid_x+1] - x[tid_x];

        Dxx[IDX2(numX,4, tid_x,0)] =  2.0/dxl/(dxl+dxu);
        Dxx[IDX2(numX,4, tid_x,1)] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[IDX2(numX,4, tid_x,2)] =  2.0/dxu/(dxl+dxu);
        Dxx[IDX2(numX,4, tid_x,3)] =  0.0;
    }
}

__global__
void setPayoff_kernel(REAL *myX, REAL *myY, REAL *myResult, const unsigned numX, const unsigned numY, const unsigned outer)
{
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_x >= numX)
        return;

    REAL strike;
    strike = 0.001*tid_outer;

    REAL payoff = max(myX[tid_x]-strike, (REAL)0.0);
    for (unsigned j=0; j < numY; ++j) {
        myResult[IDX3(outer, numX, numY, tid_outer, tid_x, j)] = payoff;
    }
}

__global__
void res_kernel(
        REAL *res,
        REAL *myResult,
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        unsigned int myXindex,
        unsigned int myYindex
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid_outer >= outer)
        return;

    res[tid_outer] = myResult[IDX3(outer, numX, numY, tid_outer, myXindex, myYindex)];
}

__global__
void rollback_explicit_x_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const unsigned numT,
        const unsigned g,
        REAL *u,
        REAL *myTimeline,
        REAL *myVarX,
        REAL *myDxx,
        REAL *myResult
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_x >= numX)
        return;

    REAL dtInv = 1.0/(myTimeline[g+1] - myTimeline[g]);

    for(int j=0; j < numY; j++) {
        REAL *myu = &u[IDX3(outer,numY,numX, tid_outer,j,tid_x)];
        REAL mymyVarX = myVarX[IDX3(outer,numX,numY, tid_outer,tid_x,j)];

        *myu = dtInv*myResult[IDX3(outer,numX,numY, tid_outer,tid_x,j)];

        if(tid_x > 0) {
            *myu += 0.5 * ( 0.5 * mymyVarX * myDxx[IDX2(numX,4, tid_x,0)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,tid_x-1,j)];
        }
        *myu  +=  0.5 * ( 0.5 * mymyVarX * myDxx[IDX2(numX,4, tid_x,1)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,tid_x,j)];
        if(tid_x < numX-1) {
            *myu += 0.5 * ( 0.5 * mymyVarX * myDxx[IDX2(numX,4, tid_x,2)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,tid_x+1,j)];
        }
    }
}

__global__
void rollback_explicit_y_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        REAL *u,
        REAL *v,
        REAL *myTimeline,
        REAL *myVarY,
        REAL *myDyy,
        REAL *myResult
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_y     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_y >= numY)
        return;

    for(int i=0; i < numX; i++) {
        REAL *myv = &v[IDX3(outer,numX,numY, tid_outer,i,tid_y)];
        REAL mymyVarY = myVarY[IDX3(outer,numX,numY, tid_outer,i,tid_y)];

        *myv = 0.0;

        if(tid_y > 0) {
            *myv += 0.5 * ( 0.5 * mymyVarY * myDyy[IDX2(numY,4, tid_y,0)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,i,tid_y-1)];
        }
        *myv  +=  0.5 * ( 0.5 * mymyVarY * myDyy[IDX2(numY,4, tid_y,1)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,i,tid_y)];
        if(tid_y < numY-1) {
            *myv += 0.5 * ( 0.5 * mymyVarY * myDyy[IDX2(numY,4, tid_y,2)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,i,tid_y+1)];
        }

        u[IDX3(outer,numY,numX, tid_outer,tid_y,i)] += *myv;
    }
}

__global__
void rollback_implicit_x_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const unsigned numZ,
        const unsigned numT,
        const unsigned g,
        REAL *myTimeline,
        REAL *myVarX,
        REAL *myDxx,
        REAL *u,
        REAL *a,
        REAL *b,
        REAL *c,
        REAL *d,
        REAL *yy
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_y     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_y >= numY)
        return;

    REAL *myA  =  &a[IDX3(outer,numZ,numZ, tid_outer,tid_y,0)];
    REAL *myB  =  &b[IDX3(outer,numZ,numZ, tid_outer,tid_y,0)];
    REAL *myC  =  &c[IDX3(outer,numZ,numZ, tid_outer,tid_y,0)];
    REAL *myYY = &yy[IDX3(outer,numZ,numZ, tid_outer,tid_y,0)];
    REAL *myU  =  &u[IDX3(outer,numY,numX, tid_outer,tid_y,0)];

    REAL dtInv = 1.0/(myTimeline[g+1] - myTimeline[g]);

    for(int i=0; i < numX; i++) {  // here a, b,c should have size [numX]
        REAL mymyVarX = myVarX[IDX3(outer,numX,numY, tid_outer,i,tid_y)];

        myA[i] =       - 0.5 * (0.5 * mymyVarX * myDxx[IDX2(numX,4, i,0)]);
        myB[i] = dtInv - 0.5 * (0.5 * mymyVarX * myDxx[IDX2(numX,4, i,1)]);
        myC[i] =       - 0.5 * (0.5 * mymyVarX * myDxx[IDX2(numX,4, i,2)]);
    }

    // here yy should have size [numX]
    tridag(myA,myB,myC,myU,numX,myU,myYY);
}

__global__
void rollback_implicit_y_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const unsigned numZ,
        const unsigned numT,
        const unsigned g,
        REAL *myTimeline,
        REAL *myVarY,
        REAL *myDyy,
        REAL *myResult,
        REAL *u,
        REAL *v,
        REAL *a,
        REAL *b,
        REAL *c,
        REAL *y,
        REAL *yy
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_x >= numX)
        return;

    REAL *myA  =  &a[IDX3(outer,numZ,numZ, tid_outer,tid_x,0)];
    REAL *myB  =  &b[IDX3(outer,numZ,numZ, tid_outer,tid_x,0)];
    REAL *myC  =  &c[IDX3(outer,numZ,numZ, tid_outer,tid_x,0)];
    REAL *myY  =  &y[IDX3(outer,numZ,numZ, tid_outer,tid_x,0)];
    REAL *myYY = &yy[IDX3(outer,numZ,numZ, tid_outer,tid_x,0)];

    REAL dtInv = 1.0/(myTimeline[g+1] - myTimeline[g]);

    for(int j=0; j < numY; j++) {  // here a, b,c should have size [numX]
        REAL mymyVarY = myVarY[IDX3(outer,numX,numY, tid_outer,tid_x,j)];

        myA[j] =       - 0.5 * (0.5 * mymyVarY * myDyy[IDX2(numY,4, j,0)]);
        myB[j] = dtInv - 0.5 * (0.5 * mymyVarY * myDyy[IDX2(numY,4, j,1)]);
        myC[j] =       - 0.5 * (0.5 * mymyVarY * myDyy[IDX2(numY,4, j,2)]);
        myY[j] = dtInv * u[IDX3(outer,numY,numX, tid_outer,j,tid_x)]
               - 0.5   * v[IDX3(outer,numX,numY, tid_outer,tid_x,j)];
    }

    // here yy should have size [numY]
    tridag(myA,myB,myC,myY,numY,&myResult[IDX3(outer,numX,numY, tid_outer, tid_x, 0)],myYY);
}

#endif
