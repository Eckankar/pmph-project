#include "ProjHelperFun.h"
#include "ProjCoreCUDACores.cu.h"
#include "Constants.h"
#include <math.h>

// CUDA error checking macros; taken from http://choorucode.com/2011/03/02/how-to-do-error-checking-in-cuda/ 
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    for(unsigned i=0;i<globs.myX.size();++i)
        for(unsigned j=0;j<globs.myY.size();++j) {
            globs.myVarX[i][j] = exp(2.0*(  beta*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
            globs.myVarY[i][j] = exp(2.0*(  alpha*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
        }
}

void setPayoff(const REAL strike, PrivGlobs& globs )
{
    for(unsigned i=0;i<globs.myX.size();++i)
    {
        REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
        for(unsigned j=0;j<globs.myY.size();++j)
            globs.myResult[i][j] = payoff;
    }
}

/*
inline void new_amazing_tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
) {
    unsigned int block_size = 128;

    // generate S_i matrices
    // TODO

    // compute S_i * S_(i-1) * S_1 matrices
    scanInc<Mat2Mult, matrix>(block_size, n, d_s, d_tmpmat);


}
*/


void   run_cuda(
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t,
                const REAL&           alpha,
                const REAL&           nu,
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
) {
    // grid
    //REAL myX[outer][numX];
    //REAL myY[outer][numY];
    //REAL myTimeline[outer][numT];
    unsigned int myXindex;
    unsigned int myYindex;

    // variable
    //REAL myResult[outer][numX][numY];

    // coeffs
    //REAL myVarX[outer][numX][numY];
    //REAL myVarY[outer][numX][numY];

    // operators
    //REAL myDxx[outer][numX][4];
    //REAL myDyy[outer][numY][4];

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    // Allocate CUDA resources
    REAL *myX_d, *myY_d, *myTimeline_d, *myDxx_d, *myDyy_d, *myResult_d, *myVarX_d, *myVarY_d, *res_d;
    CudaSafeCall( cudaMalloc((void **) &myX_d,        outer * numX * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myY_d,        outer * numY * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myTimeline_d, outer * numT * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myDxx_d,      outer * numX * 4 * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myDyy_d,      outer * numY * 4 * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myResult_d,   outer * numX * numY * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myVarX_d,     outer * numX * numY * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myVarY_d,     outer * numX * numY * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &res_d,        outer * sizeof(REAL)) );

    const dim3 block_size2 = dim3(32, 32);
    const int block_size   = block_size2.x * block_size2.y * block_size2.z;

    #define GRID(first,second) dim3(ceil((REAL)(first)/block_size2.x), ceil((REAL)(second)/block_size2.y))

    initGrid_kernel<<<ceil((REAL)outer/block_size), block_size>>>(s0, logAlpha, dx, dy, myXindex, myYindex, t,
                                  numX, numY, numT, outer, myTimeline_d, myX_d, myY_d); // 1D
    CudaCheckError();

    initOperator_kernel<<<ceil((REAL)outer/block_size), block_size>>>(myX_d, myDxx_d, outer, numX); // 1D
    CudaCheckError();
    initOperator_kernel<<<ceil((REAL)outer/block_size), block_size>>>(myY_d, myDyy_d, outer, numY); // 1D
    CudaCheckError();

    setPayoff_kernel<<<GRID(outer, numX), block_size2>>>(myX_d, myY_d, myResult_d, numX, numY, outer); // 2D
    CudaCheckError();

    // stuff
    unsigned int numZ = max(numX, numY);
    REAL u[outer][numY][numX];
    REAL v[outer][numX][numY];
    REAL a[outer][numZ][numZ];
    REAL b[outer][numZ][numZ];
    REAL c[outer][numZ][numZ];
    REAL y[outer][numZ][numZ];
    REAL yy[outer][numZ][numZ];

    REAL *u_d, *v_d, *a_d, *b_d, *c_d, *d_d, *y_d, *yy_d;
    cudaMalloc((void **) &u_d,     sizeof(u));
    cudaMalloc((void **) &v_d,     sizeof(v));
    cudaMalloc((void **) &a_d,     sizeof(a));
    cudaMalloc((void **) &b_d,     sizeof(b));
    cudaMalloc((void **) &c_d,     sizeof(c));
    cudaMalloc((void **) &y_d,     sizeof(y));
    cudaMalloc((void **) &yy_d,    sizeof(yy));

    for (int j = numT-2; j>=0; --j) {
        updateParams_large_kernel<<<GRID(outer, numX), block_size2>>>(j, alpha, beta, nu, outer, numX, numY,
                                                numT, myX_d, myY_d, myVarX_d, myVarY_d, myTimeline_d); // 2D

        rollback_explicit_x_kernel<<<GRID(outer, numX), block_size2>>>(outer, numX, numY, numT, u_d, myTimeline_d,
                                                 myVarX_d, myDxx_d, myResult_d); // 2D
        rollback_explicit_y_kernel<<<GRID(outer, numY), block_size2>>>(outer, numX, numY, u_d, v_d, myTimeline_d,
                                                 myVarX_d, myDxx_d, myResult_d); // 2D
        rollback_implicit_x_kernel<<<GRID(outer, numY), block_size2>>>(outer, numX, numY, numZ, numT, myTimeline_d,
                                                 myVarX_d, myDxx_d, u_d, a_d, b_d, c_d, d_d,
                                                 yy_d); // 2D
        rollback_implicit_y_kernel<<<GRID(outer, numX), block_size2>>>(outer, numX, numY, numZ, numT, myTimeline_d,
                                                 myVarY_d, myDyy_d, myResult_d, u_d, v_d, a_d,
                                                 b_d, c_d, y_d, yy_d); // 2D
    }

    res_kernel<<<ceil((REAL)outer/block_size), block_size>>>(res_d, myResult_d, outer, numX, numY, myXindex, myYindex);
    cudaMemcpy(res, res_d, outer * sizeof(REAL), cudaMemcpyDeviceToHost);

    // XXX: free everything maybe
}

//#endif // PROJ_CORE_ORIG
