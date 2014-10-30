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
          REAL   *uu,   // size [n] temporary
    const int stride
) {
    int    i, offset;
    REAL   beta;

    u[0*stride]  = r[0*stride];
    uu[0*stride] = b[0*stride];

    for(i=1; i<n; i++) {
        beta  = a[i*stride] / uu[(i-1)*stride];

        uu[i*stride] = b[i*stride] - beta*c[(i-1)*stride];
        u[i*stride]  = r[i*stride] - beta*u[(i-1)*stride];
    }

#if 1
    // X) this is a backward recurrence
    u[(n-1)*stride] = u[(n-1)*stride] / uu[(n-1)*stride];
    for(i=n-2; i>=0; i--) {
        u[i*stride] = (u[i*stride] - c[i*stride]*u[(i+1)*stride]) / uu[i*stride];
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
        unsigned int numX,
        unsigned int numY,
        unsigned int numT,
        REAL *myX,
        REAL *myY,
        REAL *myVarX,
        REAL *myVarY,
        REAL *myTimeline
) {
    unsigned int tid_y = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_x >= numX || tid_y >= numY)
        return;

    REAL x        = myX[tid_x],
         y        = myY[tid_y],
         timeline = myTimeline[g];

    myVarX[IDX2(numX,numY, tid_x,tid_y)]
        = exp(2.0 * (beta * log(x) + y - 0.5*nu*nu*timeline));

    myVarY[IDX2(numX,numY, tid_x,tid_y)]
        = exp(2.0 * (alpha * log(x) + y - 0.5*nu*nu*timeline));

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
        Dxx[IDX2(4,numX, 0,tid_x)] =  0.0;
        Dxx[IDX2(4,numX, 1,tid_x)] =  0.0;
        Dxx[IDX2(4,numX, 2,tid_x)] =  0.0;
        Dxx[IDX2(4,numX, 3,tid_x)] =  0.0;
    } else {
        REAL dxl = x[tid_x] - x[tid_x-1];
        REAL dxu = x[tid_x+1] - x[tid_x];

        Dxx[IDX2(4,numX, 0,tid_x)] =  2.0/dxl/(dxl+dxu);
        Dxx[IDX2(4,numX, 1,tid_x)] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[IDX2(4,numX, 2,tid_x)] =  2.0/dxu/(dxl+dxu);
        Dxx[IDX2(4,numX, 3,tid_x)] =  0.0;
    }
}

__global__
void setPayoff_kernel(REAL *myX, REAL *myY, REAL *myResult, const unsigned numX, const unsigned numY, const unsigned numZ, const unsigned outer)
{
    unsigned int tid_y = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_x >= numX || tid_y >= numY)
        return;

    REAL x = myX[tid_x];

    for (unsigned j=0; j < outer; ++j) {
        REAL strike = 0.001*j;
        REAL payoff = max(x-strike, (REAL)0.0);

        myResult[IDX3(outer, numZ, numZ, j, tid_x, tid_y)] = payoff;
    }
}

__global__
void res_kernel(
        REAL *res,
        REAL *myResult,
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const unsigned numZ,
        unsigned int myXindex,
        unsigned int myYindex
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid_outer >= outer)
        return;

    res[tid_outer] = myResult[IDX3(outer, numZ, numZ, tid_outer, myXindex, myYindex)];
}

__global__
void rollback_explicit_x_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const unsigned numT,
        const unsigned numZ,
        const unsigned g,
        REAL *u,
        REAL *myTimeline,
        REAL *myVarX,
        REAL *myDxx,
        REAL *myResult
) {
    unsigned int tid_y = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_y >= numY || tid_x >= numX)
        return;

    REAL dtInv = 1.0/(myTimeline[g+1] - myTimeline[g]);
    REAL mymyVarX = myVarX[IDX2(numX,numY, tid_x,tid_y)];

    for(int j=0; j < outer; j++) {
        REAL *myu = &u[IDX3(outer,numZ,numZ, j,tid_x,tid_y)];

        *myu = dtInv*myResult[IDX3(outer,numZ,numZ, j,tid_x,tid_y)];

        if(tid_x > 0) {
            *myu += 0.5 * ( 0.5 * mymyVarX * myDxx[IDX2(4,numX, 0,tid_x)] )
                        * myResult[IDX3(outer,numZ,numZ, j,tid_x-1,tid_y)];
        }
        *myu  +=  0.5 * ( 0.5 * mymyVarX * myDxx[IDX2(4,numX, 1,tid_x)] )
                        * myResult[IDX3(outer,numZ,numZ, j,tid_x,tid_y)];
        if(tid_x < numX-1) {
            *myu += 0.5 * ( 0.5 * mymyVarX * myDxx[IDX2(4,numX, 2,tid_x)] )
                        * myResult[IDX3(outer,numZ,numZ, j,tid_x+1,tid_y)];
        }
    }
}

__global__
void rollback_explicit_y_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const unsigned numZ,
        REAL *u,
        REAL *v,
        REAL *myTimeline,
        REAL *myVarY,
        REAL *myDyy,
        REAL *myResult
) {
    unsigned int tid_y = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_y >= numY || tid_x >= numX)
        return;


    for(int i=0; i < outer; i++) {
        REAL *myv = &v[IDX3(outer,numZ,numZ, i,tid_x,tid_y)];
        REAL mymyVarY = myVarY[IDX2(numX,numY, tid_x,tid_y)];

        *myv = 0.0;

        if(tid_y > 0) {
            *myv += 0.5 * ( 0.5 * mymyVarY * myDyy[IDX2(4,numY, 0,tid_y)] )
                        * myResult[IDX3(outer,numZ,numZ, i,tid_x,tid_y-1)];
        }
        *myv  +=  0.5 * ( 0.5 * mymyVarY * myDyy[IDX2(4,numY, 1,tid_y)] )
                        * myResult[IDX3(outer,numZ,numZ, i,tid_x,tid_y)];
        if(tid_y < numY-1) {
            *myv += 0.5 * ( 0.5 * mymyVarY * myDyy[IDX2(4,numY, 2,tid_y)] )
                        * myResult[IDX3(outer,numZ,numZ, i,tid_x,tid_y+1)];
        }

        u[IDX3(outer,numZ,numZ, i,tid_x,tid_y)] += *myv;
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
        REAL *a,
        REAL *b,
        REAL *c
) {
    unsigned int tid_y = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_x >= numX || tid_y >= numY)
        return;

    REAL dtInv    = 1.0/(myTimeline[g+1] - myTimeline[g]);
    REAL mymyVarX = myVarX[IDX2(numX,numY, tid_x,tid_y)];

    REAL myDxx0 = myDxx[IDX2(4,numX, 0,tid_x)];
    REAL myDxx1 = myDxx[IDX2(4,numX, 1,tid_x)];
    REAL myDxx2 = myDxx[IDX2(4,numX, 2,tid_x)];

    REAL a_new =       - 0.5 * (0.5 * mymyVarX * myDxx0);
    REAL b_new = dtInv - 0.5 * (0.5 * mymyVarX * myDxx1);
    REAL c_new =       - 0.5 * (0.5 * mymyVarX * myDxx2);

    for(int i=0; i < outer; i++) {  // here a, b,c should have size [numX]
        a[IDX3(outer,numZ,numZ, i,tid_x,tid_y)] = a_new;
        b[IDX3(outer,numZ,numZ, i,tid_x,tid_y)] = b_new;
        c[IDX3(outer,numZ,numZ, i,tid_x,tid_y)] = c_new;
    }
}

__global__
void rollback_implicit_x_part2_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        const unsigned numZ,
        REAL *u,
        REAL *a,
        REAL *b,
        REAL *c,
        REAL *yy
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_y     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_y >= numY)
        return;

    REAL *myA  =  &a[IDX3(outer,numZ,numZ, tid_outer,0,tid_y)];
    REAL *myB  =  &b[IDX3(outer,numZ,numZ, tid_outer,0,tid_y)];
    REAL *myC  =  &c[IDX3(outer,numZ,numZ, tid_outer,0,tid_y)];
    REAL *myYY = &yy[IDX3(outer,numZ,numZ, tid_outer,0,tid_y)];
    REAL *myU  =  &u[IDX3(outer,numZ,numZ, tid_outer,0,tid_y)];

    // here yy should have size [numX]
    tridag(myA,myB,myC,myU,numX,myU,myYY,numZ);
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
        REAL *y
) {
    unsigned int tid_y = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_y >= numY || tid_x >= numX)
        return;

    REAL dtInv = 1.0/(myTimeline[g+1] - myTimeline[g]);

    REAL myDyy0 = myDyy[IDX2(4,numY, 0,tid_y)];
    REAL myDyy1 = myDyy[IDX2(4,numY, 1,tid_y)];
    REAL myDyy2 = myDyy[IDX2(4,numY, 2,tid_y)];

    REAL mymyVarY = myVarY[IDX2(numX,numY, tid_x,tid_y)];

    REAL a_new =       - 0.5 * (0.5 * mymyVarY * myDyy0);
    REAL b_new = dtInv - 0.5 * (0.5 * mymyVarY * myDyy1);
    REAL c_new =       - 0.5 * (0.5 * mymyVarY * myDyy2);

    for(int j=0; j < outer; j++) {  // here a, b,c should have size [numX]
        a[IDX3(outer,numZ,numZ, j,tid_x,tid_y)] = a_new;
        b[IDX3(outer,numZ,numZ, j,tid_x,tid_y)] = b_new;
        c[IDX3(outer,numZ,numZ, j,tid_x,tid_y)] = c_new;

        y[IDX3(outer,numZ,numZ, j,tid_x,tid_y)]
            = dtInv * u[IDX3(outer,numZ,numZ, j,tid_x,tid_y)]
            - 0.5   * v[IDX3(outer,numZ,numZ, j,tid_x,tid_y)];
    }
}

__global__
void rollback_implicit_y_part2_kernel(
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
    unsigned int tid_x     = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_outer = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_x >= numX)
        return;

    REAL *myA  =  &a[IDX3(outer,numZ,numZ, tid_outer,0,tid_x)];
    REAL *myB  =  &b[IDX3(outer,numZ,numZ, tid_outer,0,tid_x)];
    REAL *myC  =  &c[IDX3(outer,numZ,numZ, tid_outer,0,tid_x)];
    REAL *myY  =  &y[IDX3(outer,numZ,numZ, tid_outer,0,tid_x)];
    REAL *myYY = &yy[IDX3(outer,numZ,numZ, tid_outer,0,tid_x)];

    // here yy should have size [numY]
    tridag(myA,myB,myC,myY,numY,&myResult[IDX3(outer,numZ,numZ, tid_outer, 0, tid_x)],myYY,numZ);
}

#endif
