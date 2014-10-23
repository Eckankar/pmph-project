#ifndef PROJCORECUDACORES_CU_H_85E8271D
#define PROJCORECUDACORES_CU_H_85E8271D

#define IDX2(DIMX, DIMY, X, Y)           (X*DIMY + Y)
#define IDX3(DIMX, DIMY, DIMZ, X, Y, Z)  (X*DIMY*DIMZ + Y*DIMZ + Z)

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
        REAL x        = myX[IDX2(max_outer, numX, tid_outer, tid_x)],
             y        = myX[IDX2(max_outer, numY, tid_outer, j)],
             timeline = myTimeline[IDX2(max_outer, numT, tid_outer, g)];

        myVarX[IDX3(max_outer, numX, numY, tid_outer, tid_x, j)]
            = exp(2.0 * (beta * log(x) + y - 0.5*nu*nu*timeline));

        myVarY[IDX3(max_outer, numX, numY, tid_outer, tid_x, j)]
            = exp(2.0 * (alpha * log(x) + y - 0.5*nu*nu*timeline));
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
        const unsigned outer,
        REAL *myTimeline,
        REAL *myX,
        REAL *myY
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid_outer >= outer)
        return;

    for (unsigned i=0;i<numT;++i) {
        myTimeline[IDX2(outer, numT, tid_outer, i)] = t*i/(numT-1); // [tid_outer][i]
    }

    for(unsigned i=0;i<numX;++i) {
        //globs.myX[i] =
        myX[IDX2(outer, numX, tid_outer, i)] = i*dx - myXindex*dx + s0;
    }

    for(unsigned i=0;i<numY;++i) {
        //globs.myY[i] =
        myY[IDX2(outer, numY, tid_outer, i)] = i*dy - myYindex*dy + logAlpha;
    }
}

__global__
void initOperator_kernel(REAL *x, REAL *Dxx, const unsigned outer, const unsigned max_x) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid_outer >= outer)
        return;

    REAL dxl, dxu;

    //  lower boundary
    dxl      =  0.0;
    dxu      =  x[IDX2(outer, numX, tid_outer, 1)] - x[IDX2(outer, numX, tid_outer, 0)];

    Dxx[IDX3(outer, numX, 4, tid_outer, 0, 0)] =  0.0;
    Dxx[IDX3(outer, numX, 4, tid_outer, 0, 1)] =  0.0;
    Dxx[IDX3(outer, numX, 4, tid_outer, 0, 2)] =  0.0;
    Dxx[IDX3(outer, numX, 4, tid_outer, 0, 3)] =  0.0;

    //  standard case
    for(unsigned i=1;i<max_x-1;i++)
    {
        dxl      = x[IDX2(outer, numX, tid_outer, i)]   - x[IDX2(outer, numX, tid_outer, i-1)];
        dxu      = x[IDX2(outer, numX, tid_outer, i+1)] - x[IDX2(outer, numX, tid_outer, i)];

        Dxx[IDX3(outer, numX, 4, tid_outer, i, 0)] =  2.0/dxl/(dxl+dxu);
        Dxx[IDX3(outer, numX, 4, tid_outer, i, 1)] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[IDX3(outer, numX, 4, tid_outer, i, 2)] =  2.0/dxu/(dxl+dxu);
        Dxx[IDX3(outer, numX, 4, tid_outer, i, 3)] =  0.0;
    }

    //  upper boundary
    dxl        =  x[IDX2(outer, numX, tid_outer, n-1)] - x[IDX2(outer, numX, tid_outer, max_x-2)];
    dxu        =  0.0;

    Dxx[IDX3(outer, numX, 4, tid_outer, n-1, 0)] = 0.0;
    Dxx[IDX3(outer, numX, 4, tid_outer, n-1, 1)] = 0.0;
    Dxx[IDX3(outer, numX, 4, tid_outer, n-1, 2)] = 0.0;
    Dxx[IDX3(outer, numX, 4, tid_outer, n-1, 3)] = 0.0;
}

__global__
void setPayoff_kernel(REAL *myX, REAL *myY, const unsigned maxX, const unsigned maxY, const unsigned outer)
{
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_x >= max_x)
        return;

    REAL strike;
    strike = 0.001*tid_x;

    REAL payoff = max(myX[IDX2(outer, numX, tid_outer, tid_x)]-strike, (REAL)0.0);
    for (unsigned j=0; j < maxY; ++j) {
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

    REAL dtInv = 1.0/(myTimeline[IDX2(outer,numT,tid_outer,tid_outer+1)]-myTimeline[IDX2(outer, numT, tid_outer, tid_outer)]);

    for(j=0; j < numY; j++) {
        REAL *myu = &u[IDX3(outer,numY,numX, tid_outer,j,tid_x)];
        REAL mymyVarX = myVarX[IDX3(outer,numX,numY, tid_outer,tid_x,j)];

        *myu = dtInv*myResult[IDX3(outer,numX,numY, tid_outer,tid_x,j)];

        if(i > 0) {
            *myu += 0.5 * ( 0.5 * mymyVarX * myDxx[IDX3(outer,numX,4, tid_outer,tid_x,0)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,tid_x-1,j)];
        }
        *myu  +=  0.5 * ( 0.5 * mymyVarX * myDxx[IDX3(outer,numX,4, tid_outer,tid_x,1)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,tid_x,j)];
        if(tid_x < numX-1) {
            *myu += 0.5 * ( 0.5 * mymyVarX * myDxx[IDX3(outer,numX,4, tid_outer,tid_x,2)] )
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

    for(i=0; i < numX; i++) {
        REAL *myv = &v[IDX3(outer,numY,numX, tid_outer,tid_y,i)];
        REAL mymyVarY = myVarY[IDX3(outer,numX,numY, tid_outer,tid_y,i)];

        *myv = 0.0f;

        if(tid_y > 0) {
            *myv += 0.5 * ( 0.5 * mymyVarY * myDyy[IDX3(outer,numX,4, tid_outer,tid_y,0)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,i,tid_y-1)];
        }
        *myv  +=  0.5 * ( 0.5 * mymyVarY * myDyy[IDX3(outer,numX,4, tid_outer,tid_y,1)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,i,tid_y)];
        if(tid_y < numY-1) {
            *myv += 0.5 * ( 0.5 * mymyVarY * myDyy[IDX3(outer,numX,4, tid_outer,tid_y,2)] )
                        * myResult[IDX3(outer,numX,numY, tid_outer,i,tid_y+1)];
        }

        u[IDX3(outer,numY,numX, tid_outer,i,tid_y)] += *myv;
    }
}

__global__
void rollback_implicit_x_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        REAL *myTimeline,
        REAL *myVarX,
        REAL *myDxx,
        REAL *u
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_y     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_y >= numY)
        return;

    REAL dtInv = 1.0/(myTimeline[IDX2(outer,numT,tid_outer,tid_outer+1)]-myTimeline[IDX2(outer, numT, tid_outer, tid_outer)]);

    unsigned int numZ = max(numX, numY);

    REAL a[numZ], b[numZ], c[numZ];
    REAL yy[numZ];

    for(i=0; i < numX; i++) {  // here a, b,c should have size [numX]
        REAL mymyVarX = myVarX[IDX3(outer,numX,numY, tid_outer,tid_y,j)];

        a[i] =       - 0.5 * (0.5 * mymyVarX * myDxx[IDX3(outer,numX,4, tid_outer,i,0)]);
        b[i] = dtInv - 0.5 * (0.5 * mymyVarX * myDxx[IDX3(outer,numX,4, tid_outer,i,1)]);
        c[i] =       - 0.5 * (0.5 * mymyVarX * myDxx[IDX3(outer,numX,4, tid_outer,i,2)]);
    }

    // here yy should have size [numX]
    // TODO:
    //tridag(a,b,c,u[j],numX,u[j],yy);
}

__global__
void rollback_implicit_y_kernel(
        const unsigned outer,
        const unsigned numX,
        const unsigned numY,
        REAL *myTimeline,
        REAL *myVarY,
        REAL *myDyy,
        REAL *myResult,
        REAL *u_d,
        REAL *v_d
) {
    unsigned int tid_outer = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid_x     = blockIdx.y*blockDim.y + threadIdx.y;

    if (tid_outer >= outer || tid_x >= numX)
        return;

    REAL dtInv = 1.0/(myTimeline[IDX2(outer,numT,tid_outer,tid_outer+1)]-myTimeline[IDX2(outer, numT, tid_outer, tid_outer)]);

    unsigned int numZ = max(numX, numY);

    REAL a[numZ], b[numZ], c[numZ], y[numZ];
    REAL yy[numZ];

    for(j=0; j < numY; j++) {  // here a, b,c should have size [numX]
        REAL mymyVarX = myVarX[IDX3(outer,numX,numY, tid_outer,tid_x,j)];

        a[j] =       - 0.5 * (0.5 * mymyVarY * myDyy[IDX3(outer,numX,4, tid_outer,j,0)]);
        b[j] = dtInv - 0.5 * (0.5 * mymyVarY * myDyy[IDX3(outer,numX,4, tid_outer,j,1)]);
        c[j] =       - 0.5 * (0.5 * mymyVarY * myDyy[IDX3(outer,numX,4, tid_outer,j,2)]);

        for(j=0;j<numY;j++) {
            y[j] = dtInv * u[IDX3(outer,numY,numX, tid_outer,j,tid_x)]
                 - 0.5   * v[IDX3(outer,numX,numY, tid_outer,tid_x,j)];
        }
    }

    // here yy should have size [numY]
    // TODO:
    //tridag(a,b,c,y,numY,myResult[IDX2(outer,numX,numY, tid_outer, tid_x, 0)],yy);
}

#endif
