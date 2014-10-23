#include "ProjHelperFun.h"
#include "ProjCoreCUDACores.cu.h"
#include "Constants.h"

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


inline void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
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


void
rollback( const unsigned g, PrivGlobs& globs ) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]

    //    explicit x
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            u[j][i] = dtInv*globs.myResult[i][j];

            if(i > 0) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] )
                            * globs.myResult[i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * globs.myResult[i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                            * globs.myResult[i+1][j];
            }
        }
    }

    //    explicit y
    for(j=0;j<numY;j++)
    {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                         *  globs.myResult[i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                         *  globs.myResult[i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                         *  globs.myResult[i][j+1];
            }
            u[j][i] += v[i][j];
        }
    }

    //    implicit x
    for(j=0;j<numY;j++) {
        vector<REAL> a(numZ), b(numZ), c(numZ);     // [max(numX,numY)]
        vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a[i] =         - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            b[i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            c[i] =         - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
        }
        // here yy should have size [numX]
        tridag(a,b,c,u[j],numX,u[j],yy);
    }

    //    implicit y
    for(i=0;i<numX;i++) {
        vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)]
        vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =         - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            b[j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            c[j] =         - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
        }

        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        tridag(a,b,c,y,numY,globs.myResult[i],yy);
    }
}

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
    REAL myX[outer][numX];
    REAL myY[outer][numY];
    REAL myTimeline[outer][numT];
    unsigned int myXindex;
    unsigned int myYindex;

    // variable
    REAL myResult[outer][numX][numY];

    // coeffs
    REAL myVarX[outer][numX][numY];
    REAL myVarY[outer][numX][numY];

    // operators
    REAL myDxx[outer][numX][4];
    REAL myYxx[outer][numY][4];

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    // Allocate CUDA resources
    REAL *myX_d, *myY_d, *myTimeline_d, *myDxx_d, *myDyy_d, *myResult_d, *myVarX_d, *myVarY_d;
    cudaMalloc((void **) &myX_d,        sizeof(myX));
    cudaMalloc((void **) &myY_d,        sizeof(myY));
    cudaMalloc((void **) &myTimeline_d, sizeof(myTimeline));
    cudaMalloc((void **) &myDxx_d,      sizeof(myDxx));
    cudaMalloc((void **) &myDyy_d,      sizeof(myDyy));
    cudaMalloc((void **) &myResult_d,   sizeof(myResult));
    cudaMalloc((void **) &myVarX_d,     sizeof(myVarX));
    cudaMalloc((void **) &myVarY_d,     sizeof(myVarY));
    cudaMalloc((void **) &res_d,        sizeof(REAL * outer));

    initGrid_kernel<<<foo, bar>>>(s0, logAlpha, dx, dy, myXindex, myYindex, t,
                                  numX, numY, numT, outer, myTimeline_d, myX_d, myY_d); // 1D
    initOperator_kernel<<<foo, bar>>>(myX_d, myDxx_d, outer, numX); // 1D
    initOperator_kernel<<<foo, bar>>>(myY_d, myDyy_d, outer, numY); // 1D
    setPayoff_kernel<<<foo, bar>>>(myX_d, myY_d, maxX, maxY, outer); // 2D

    for (int j = numT-2; j>=0; --j) {
        updateParams_large_kernel<<<bla, bla>>>(j, alpha, beta, nu, outer, numX, numY,
                                                numT, myX_d, myY_d, myVarX_d, myVarY_d, myTimeline); // 2D

        for (unsigned i = 0; i < outer; ++ i ) {
            rollback(j, globs[i]);
        }
    }

    res_kernel<<<foo, bar>>>(res_d, myResult_d, outer, numX, numY, myXindex, myYindex);
}

//#endif // PROJ_CORE_ORIG
