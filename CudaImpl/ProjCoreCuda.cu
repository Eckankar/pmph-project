#include "ProjHelperFun.h"
#include "ProjCoreCUDACores.cu.h"
#include "CudaUtilProj.cu.h"
#include "Constants.h"

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
            if (i == 52 && j == 253) {
                printf("non-cuda: %.10f %.10f %.10f\n", globs.myX[i], globs.myY[j], globs.myTimeline[g]);
                printf("non-cuda: %.10f %.10f\n", log(globs.myX[i]), globs.myVarY[i][j]);

            }
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
    //REAL myX[numX];
    //REAL myY[numY];
    //REAL myTimeline[numT];
    unsigned int myXindex;
    unsigned int myYindex;

    // variable
    //REAL myResult[outer][numX][numY];

    // coeffs
    //REAL myVarX[outer][numX][numY];
    //REAL myVarY[outer][numX][numY];

    // operators
    //REAL myDxx[numX][4];
    //REAL myDyy[numY][4];

#if DO_DEBUG
    REAL testMatrix[2][3][4] = {
        { {1,2,3,4},
          {5,6,7,8},
          {9,10,11,12} },
        { {13,14,15,16},
          {17,18,19,20},
          {21,22,23,24} }
    };
    REAL expectedMatrix[2][4][3] = {
        { {1,5,9},
          {2,6,10},
          {3,7,11},
          {4,8,12} },
        { {13,17,21},
          {14,18,22},
          {15,19,23},
          {16,20,24} }
    };
    REAL testMatrix_t[2][4][3] = {0};

    REAL *testMatrix_d, *testMatrix_t_d;
    CudaSafeCall( cudaMalloc( (void **) &testMatrix_d, 2*3*4*sizeof(REAL) ) );
    CudaSafeCall( cudaMalloc( (void **) &testMatrix_t_d, 2*4*3*sizeof(REAL) ) );
    cudaMemcpy(testMatrix_d, testMatrix, 2*3*4*sizeof(REAL), cudaMemcpyHostToDevice);

    transpose3d(testMatrix_d, testMatrix_t_d, 2, 3, 4);
    cudaMemcpy(testMatrix_t, testMatrix_t_d, 2*3*4*sizeof(REAL), cudaMemcpyDeviceToHost);

    printf("Got:\n");
    for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 3; k++) {
        printf("%.0f ", testMatrix_t[i][j][k]);
    }
        printf("\n");
    }
        printf("\n");
    }

    printf("Expected:\n");
    for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 3; k++) {
        printf("%.0f ", expectedMatrix[i][j][k]);
    }
        printf("\n");
    }
        printf("\n");
    }
    printf("Matrix verificeret\n");
#endif

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    unsigned int numZ = max(numX, numY);

    // Allocate CUDA resources
    REAL *myX_d, *myY_d, *myTimeline_d, *myDxx_d, *myDyy_d, *myResult_d, *myVarX_d, *myVarY_d, *res_d;
    CudaSafeCall( cudaMalloc((void **) &myX_d,        numX * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myY_d,        numY * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myTimeline_d, numT * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myDxx_d,      numX * 4 * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myDyy_d,      numY * 4 * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myResult_d,   outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myVarX_d,     numX * numY * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myVarY_d,     numX * numY * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &res_d,        outer * sizeof(REAL)) );

    // Allocate transposed resources
    REAL *myDxx_t_d, *myDyy_t_d, *myResult_t_d;
    CudaSafeCall( cudaMalloc((void **) &myDxx_t_d,      numX * 4 * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myDyy_t_d,      numY * 4 * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &myResult_t_d,   outer * numZ * numZ * sizeof(REAL)) );
    const dim3 block_size2 = dim3(32, 32);
    const dim3 block_size3 = dim3(8, 8, 8);
    const int block_size   = block_size2.x * block_size2.y * block_size2.z;

    #define GRID(first,second) dim3(ceil((REAL)(first)/block_size2.x), ceil((REAL)(second)/block_size2.y))
    #define GRID3(first,second,third) dim3(ceil((REAL)(first)/block_size3.x), ceil((REAL)(second)/block_size3.y), ceil((REAL)(third)/block_size3.y))
    //CudaSafeCall(cudaMemcpy(myX_d, myX, outer * numX * sizeof(REAL), cudaMemcpyHostToDevice));
    //CudaSafeCall(cudaMemcpy(myY_d,        myY, outer * numY * sizeof(REAL), cudaMemcpyHostToDevice));
    //CudaSafeCall(cudaMemcpy(myTimeline_d, myTimeline, outer * numT * sizeof(REAL), cudaMemcpyHostToDevice));

    unsigned int maxXYT = max(numX, max(numY, numT));

    initGrid_kernel<<<ceil((REAL)maxXYT/block_size), block_size>>>(s0, logAlpha, dx, dy, myXindex, myYindex, t,
                                  numX, numY, numT, myTimeline_d, myX_d, myY_d); // 1D
    CudaCheckError();

#if DO_DEBUG
    PrivGlobs globs(numX, numY, numT);
    initGrid(s0, alpha, nu, t, numX, numY, numT, globs);

    REAL *myX, *myY, *myTimeline;
    myX = (REAL*) malloc(numX * sizeof(REAL));
    myY = (REAL*) malloc(numY * sizeof(REAL));
    myTimeline = (REAL*) malloc(numT * sizeof(REAL));

    cudaMemcpy(myX, myX_d, numX * sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(myY, myY_d, numY * sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(myTimeline, myTimeline_d, numT * sizeof(REAL), cudaMemcpyDeviceToHost);

    for (int x = 0; x < numX; ++x) {
        REAL x1 = globs.myX[x];
        REAL x2 = myX[x];
        if (abs(x1-x2) >= 1e-7) {
            printf("myX(%d), %.14f, %.14f, %.14f\n", x, abs(x1-x2), x1, x2);
        }
    }

    for (int i = 0; i < numX; i++) {
        globs.myX[i] = myX[i];
    }
    for (int i = 0; i < numY; i++) {
        globs.myY[i] = myY[i];
    }
    for (int i = 0; i < numT; i++) {
        globs.myTimeline[i] = myTimeline[i];
    }

    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);
#endif

    initOperator_kernel<<<ceil((REAL)numX/block_size), block_size>>>(myX_d, myDxx_t_d, numX); // 1D
    CudaCheckError();

    initOperator_kernel<<<ceil((REAL)numY/block_size), block_size>>>(myY_d, myDyy_t_d, numY); // 1D
    CudaCheckError();

#if DO_DEBUG
    transpose<REAL,32>(myDxx_t_d, myDxx_d, 4, numX);
    transpose<REAL,32>(myDyy_t_d, myDyy_d, 4, numY);

    REAL *myDxx;
    myDxx = (REAL*) malloc(numX * 4 * sizeof(REAL));

    cudaMemcpy(myDxx, myDxx_d, numX * 4 * sizeof(REAL), cudaMemcpyDeviceToHost);

    for (int x = 0; x < numX; ++x) {
    for (int i = 0; i < 4; ++i) {
        REAL x1 = globs.myDxx[x][i];
        REAL x2 = myDxx[IDX2(numX,4, x,i)];
        if (abs(x1-x2) >= 1e-10) {
            printf("myDxx(%d,%d), %.14f, %.14f, %.14f\n", x, i, abs(x1-x2), x1, x2);
        }
    }
    }

    printf("Initoperator checked.\n");
#endif

    setPayoff_kernel<<<GRID(numY, numX), block_size2>>>(myX_d, myY_d, myResult_d, numX, numY, numZ, outer); // 2D

#if DO_DEBUG
    REAL *myResult;
    myResult = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
    cudaMemcpy(myResult, myResult_d, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
    CudaCheckError();

    setPayoff(0.001 * 7, globs);
    for (int x = 0; x < numX; ++x) {
    for (int y = 0; y < numY; ++y) {
        REAL x1 = globs.myResult[x][y];
        REAL x2 = myResult[IDX3(outer,numX,numY, 7,x,y)];
        if (abs(x1-x2) >= 1e-10) {
            printf("myResult(%d,%d,%d), %.14f, %.14f, %.14f\n", 7, x, y, abs(x1-x2), x1, x2);
        }
    }
    }

    printf("setPayoff checked.\n");
#endif

    // stuff
   //REAL u[outer][numY][numX];
    //REAL v[outer][numX][numY];
    //REAL a[outer][numZ][numZ];
    //REAL b[outer][numZ][numZ];
    //REAL c[outer][numZ][numZ];
    //REAL y[outer][numZ][numZ];
    //REAL yy[outer][numZ][numZ];

    REAL *u_d, *v_d, *a_d, *b_d, *c_d, *y_d, *yy_d;
    CudaSafeCall( cudaMalloc((void **) &u_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &v_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &a_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &b_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &c_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &y_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &yy_d,    outer * numZ * numZ * sizeof(REAL)) );

    // Transposed
    REAL *u_t_d, *a_t_d, *b_t_d, *c_t_d, *y_t_d;
    CudaSafeCall( cudaMalloc((void **) &u_t_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &a_t_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &b_t_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &c_t_d,     outer * numZ * numZ * sizeof(REAL)) );
    CudaSafeCall( cudaMalloc((void **) &y_t_d,     outer * numZ * numZ * sizeof(REAL)) );


    for (int j = numT-2; j>=0; --j) {
#if DO_DEBUG
        cudaDeviceSynchronize();
        printf("time step %d\n", j);
#endif

        updateParams_large_kernel<<<GRID(numY, numX), block_size2>>>(j, alpha, beta, nu, numX, numY,
                                                numT, myX_d, myY_d, myVarX_d, myVarY_d, myTimeline_d); // 2D

#if DO_DEBUG
        if (j == numT-2) {
            REAL *myVarX, *myVarY;
            myVarX = (REAL*) malloc(numX * numY * sizeof(REAL));
            myVarY = (REAL*) malloc(numX * numY * sizeof(REAL));
            cudaMemcpy(myVarX, myVarX_d, numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
            cudaMemcpy(myVarY, myVarY_d, numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
            CudaCheckError();

            updateParams(j, alpha, beta, nu, globs);

            for (int x = 52; x < 53; ++x) {
            for (int y = 253; y < 254; ++y) {
                REAL x1 = globs.myVarX[x][y];
                REAL x2 = myVarX[IDX2(numX,numY, x,y)];
                if (abs(x1-x2) >= 1e-7) {
                    printf("myVarX(%d,%d), %.14f, %.14f, %.14f\n", x, y, abs(x1-x2), x1, x2);
                }

                x1 = globs.myVarY[x][y];
                x2 = myVarY[IDX2(numX,numY, x,y)];
                if (abs(x1-x2) >= 1e-7) {
                    printf("myVarY(%d,%d), %.14f, %.14f, %.14f\n", x, y, abs(x1-x2), x1, x2);
                }
            }
            }

            printf("updateParams checked.\n");
        }
#endif

        rollback_explicit_x_kernel<<<GRID(numY, numX), block_size2>>>(outer, numX, numY, numT, numZ, j, u_t_d,
                myTimeline_d, myVarX_d, myDxx_t_d, myResult_d); // 2D

        rollback_explicit_y_kernel<<<GRID(numY, numX), block_size2>>>(outer, numX, numY, numZ, u_t_d, v_d,
                myTimeline_d, myVarY_d, myDyy_t_d, myResult_d); // 2D
        cudaDeviceSynchronize();
        printf("pre-first transpose\n");
        transpose3d(u_t_d, u_d, outer, numX, numY);
        cudaDeviceSynchronize();
        printf("post-first transpose\n");

        rollback_implicit_x_kernel<<<GRID(numY, numX), block_size2>>>(outer, numX, numY, numZ, numT, j,
                        myTimeline_d, myVarX_d, myDxx_t_d, a_t_d, b_t_d, c_t_d);

        cudaDeviceSynchronize();
        printf("pre-second transpose\n");
        transpose3d(a_t_d, a_d, outer, numZ, numZ);
        cudaDeviceSynchronize();
        printf("post-second transpose\n");
        transpose3d(b_t_d, b_d, outer, numZ, numZ);
        transpose3d(c_t_d, c_d, outer, numZ, numZ);

        rollback_implicit_x_part2_kernel<<<GRID(outer, numY), block_size2>>>(outer, numX, numY, numZ, u_t_d,
                a_t_d, b_t_d, c_t_d, yy_d); // 2D

        transpose3d(u_d, u_t_d, outer, numY, numX);


        rollback_implicit_y_kernel<<<GRID(numY, numX), block_size2>>>(outer, numX, numY, numZ, numT, j,
                myTimeline_d, myVarY_d, myDyy_t_d, myResult_d, u_t_d, v_d, a_d, b_d, c_d, y_d); // 2D

        transpose3d(myResult_d, myResult_t_d, outer, numZ, numZ);

        rollback_implicit_y_part2_kernel<<<GRID(outer, numX), block_size2>>>(outer, numX, numY, numZ, numT, j, myTimeline_d,
                                                 myVarY_d, myDyy_d, myResult_t_d, u_t_d, v_d, a_t_d,
                                                 b_t_d, c_t_d, y_t_d, yy_d); // 2D

        transpose3d(myResult_t_d, myResult_d, outer, numZ, numZ);
    }

    printf("pre-res\n");
    res_kernel<<<ceil((REAL)outer/block_size), block_size>>>(res_d, myResult_d, outer, numX, numY, numZ, myXindex, myYindex);
    cudaDeviceSynchronize();
    printf("post-res\n");
    cudaMemcpy(res, res_d, outer * sizeof(REAL), cudaMemcpyDeviceToHost);
    printf("post-memcpy\n");

    // XXX: free everything maybe
}

//#endif // PROJ_CORE_ORIG
