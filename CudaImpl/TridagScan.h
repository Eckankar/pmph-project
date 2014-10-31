#include "Constants.h"

#if (WITH_FLOATS==0)
    struct REAL2 {
        double x;
        double y;
    };
    struct REAL4 {
        double x;
        double y;
        double z;
        double w;
    };
#else
    struct REAL2 {
        float x;
        float y;
    };
    struct REAL4 {
        float x;
        float y;
        float z;
        float w;
    };
#endif

// linear function composition operator
REAL2 linFunOp(REAL2 u, REAL2 v) {
    REAL2 res;
    res.x = (v.x + v.y*u.x);
    res.y = u.y*v.y;
    return res;
}

// scan inclusive with linFunOp
// (and neutral element {e.x = 0.0, e.y = 1.0})
void scanIncLinFun( const int            N,
                    const vector<REAL2>& inp,
                          vector<REAL2>& out ) {
    out[0].x = inp[0].x; out[0].y = inp[0].y;
    for(int i=1; i<N; i++) {
        out[i] = linFunOp(out[i-1],inp[i]);
    }   
}

// 2x2 matrix multiplication operator
inline REAL4 matMult22Op(REAL4 b, REAL4 a) {
    REAL4 res;
    res.x = a.x*b.x + a.y*b.z;    res.y = a.x*b.y + a.y*b.w; 
    res.z = a.z*b.x + a.w*b.z;    res.w = a.z*b.y + a.w*b.w;

    REAL val = 1.0/(a.x*b.x);
    res.x *= val; res.y *= val; res.w *= val; res.z *= val;
    return res;
}

// scan inclusive with matMult22Op
// (and neutral element 
// {e.x = 1.0, e.y = 0.0, e.z = 1.0, e.w = 0.0})
void scanIncMatMult( const int            N,
                     const vector<REAL4>& inp,
                           vector<REAL4>& out ) {
    out[0].x = inp[0].x; out[0].y = inp[0].y;
    out[0].z = inp[0].z; out[0].w = inp[0].w;
    for(int i=1; i<N; i++) {
        out[i] = matMult22Op(out[i-1], inp[i]);
    }   
}



void tridagScan(
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
    REAL   u0;

    {
        u0 = b[0];
        uu[0] = u0;

        vector<REAL4> inp_4tmp; inp_4tmp.resize(n);
        vector<REAL4> out_4tmp; out_4tmp.resize(n);

        inp_4tmp[0].x = u0; inp_4tmp[0].y = 0.0;
        inp_4tmp[0].w = u0; inp_4tmp[0].z = 0.0;
        for(i=1; i<n; i++) {
            inp_4tmp[i].x = b[i];  
            inp_4tmp[i].y = (0.0 - a[i]*c[i-1]); 
            inp_4tmp[i].z = 1.0; 
            inp_4tmp[i].w = 0.0;
        }
        scanIncMatMult(n, inp_4tmp, out_4tmp);
        //uu[0] = u0;
        for(int i=0; i<n; i++) {
            REAL4& cur = out_4tmp[i];
#if 1
            REAL nom   = cur.x*u0 + cur.y;
            REAL denom = cur.z*u0 + cur.w;
            
            uu[i] = nom/denom;
#else
            REAL denom = cur.z*u0+cur.w;
            REAL nom = (cur.x/denom)*u0 + (cur.y/denom);
            uu[i] = nom;
#endif
        }
    }

    {
        u0 = r[0];
        vector<REAL2> inp_2tmp; inp_2tmp.resize(n);
        vector<REAL2> out_2tmp; out_2tmp.resize(n);     
        //FORWARD PASS; neutral element (0.0, 1.0)
        inp_2tmp[0].x = 0.0; inp_2tmp[0].y = 1.0; 
        for(i=1; i<n; i++) {
            inp_2tmp[i].x = r[i]; 
            inp_2tmp[i].y = 0.0 - a[i]/uu[i-1];
        }
        scanIncLinFun(n, inp_2tmp, out_2tmp);
        for(i=0; i<n; i++) {
            u[i] = out_2tmp[i].x + u0 * out_2tmp[i].y;
        }

        //BACKWARD PASS
        u0 = u[n-1] / uu[n-1];
        inp_2tmp[0].x = 0.0; inp_2tmp[0].y = 1.0;
        for(i=1; i<n; i++) {
            inp_2tmp[i].x = u[n-1-i]/uu[n-1-i]; 
            inp_2tmp[i].y = 0.0 - c[n-1-i]/uu[n-1-i];
        }
        scanIncLinFun(n, inp_2tmp, out_2tmp);
        for(i=0; i<n; i++) {
            u[n-1-i] = out_2tmp[i].x + u0 * out_2tmp[i].y;
        }
    }    
}

