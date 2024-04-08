/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) rockit_model_expl_vde_adj_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s3[8] = {6, 1, 0, 4, 2, 3, 4, 5};

/* rockit_model_expl_vde_adj:(i0[4],i1[4],i2[2],i3[7])->(o0[6x1,4nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real *rr;
  const casadi_real *cs;
  casadi_real w0, w1, w2, w3, *w4=w+4, *w5=w+8, w6, w7, w8, w9, w10, w11, w12, w13;
  /* #0: @0 = input[0][2] */
  w0 = arg[0] ? arg[0][2] : 0;
  /* #1: @1 = cos(@0) */
  w1 = cos( w0 );
  /* #2: @2 = input[2][1] */
  w2 = arg[2] ? arg[2][1] : 0;
  /* #3: @3 = input[0][3] */
  w3 = arg[0] ? arg[0][3] : 0;
  /* #4: @4 = input[1][0] */
  casadi_copy(arg[1], 4, w4);
  /* #5: @5 = (@3*@4) */
  for (i=0, rr=w5, cs=w4; i<4; ++i) (*rr++)  = (w3*(*cs++));
  /* #6: {@3, @6, @7, NULL} = vertsplit(@5) */
  w3 = w5[0];
  w6 = w5[1];
  w7 = w5[2];
  /* #7: @8 = (@2*@6) */
  w8  = (w2*w6);
  /* #8: @1 = (@1*@8) */
  w1 *= w8;
  /* #9: @8 = sin(@0) */
  w8 = sin( w0 );
  /* #10: @9 = (@2*@3) */
  w9  = (w2*w3);
  /* #11: @8 = (@8*@9) */
  w8 *= w9;
  /* #12: @1 = (@1-@8) */
  w1 -= w8;
  /* #13: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  /* #14: @1 = cos(@0) */
  w1 = cos( w0 );
  /* #15: @8 = (@2*@1) */
  w8  = (w2*w1);
  /* #16: @0 = sin(@0) */
  w0 = sin( w0 );
  /* #17: @9 = (@2*@0) */
  w9  = (w2*w0);
  /* #18: @10 = input[2][0] */
  w10 = arg[2] ? arg[2][0] : 0;
  /* #19: @11 = tan(@10) */
  w11 = tan( w10 );
  /* #20: @12 = (@2*@11) */
  w12  = (w2*w11);
  /* #21: @13 = 0 */
  w13 = 0.;
  /* #22: @5 = vertcat(@8, @9, @12, @13) */
  rr=w5;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w12;
  *rr++ = w13;
  /* #23: @8 = dot(@5, @4) */
  w8 = casadi_dot(4, w5, w4);
  /* #24: output[0][1] = @8 */
  if (res[0]) res[0][1] = w8;
  /* #25: @2 = (@2*@7) */
  w2 *= w7;
  /* #26: @10 = cos(@10) */
  w10 = cos( w10 );
  /* #27: @10 = sq(@10) */
  w10 = casadi_sq( w10 );
  /* #28: @2 = (@2/@10) */
  w2 /= w10;
  /* #29: output[0][2] = @2 */
  if (res[0]) res[0][2] = w2;
  /* #30: @11 = (@11*@7) */
  w11 *= w7;
  /* #31: @0 = (@0*@6) */
  w0 *= w6;
  /* #32: @11 = (@11+@0) */
  w11 += w0;
  /* #33: @1 = (@1*@3) */
  w1 *= w3;
  /* #34: @11 = (@11+@1) */
  w11 += w1;
  /* #35: output[0][3] = @11 */
  if (res[0]) res[0][3] = w11;
  return 0;
}

CASADI_SYMBOL_EXPORT int rockit_model_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int rockit_model_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int rockit_model_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void rockit_model_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int rockit_model_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void rockit_model_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void rockit_model_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void rockit_model_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int rockit_model_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int rockit_model_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real rockit_model_expl_vde_adj_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* rockit_model_expl_vde_adj_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* rockit_model_expl_vde_adj_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* rockit_model_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* rockit_model_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int rockit_model_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 8;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 20;
  return 0;
}

CASADI_SYMBOL_EXPORT int rockit_model_expl_vde_adj_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 8*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 5*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 20*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
