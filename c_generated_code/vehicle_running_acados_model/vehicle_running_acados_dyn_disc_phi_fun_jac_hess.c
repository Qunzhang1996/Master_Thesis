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
  #define CASADI_PREFIX(ID) vehicle_running_acados_dyn_disc_phi_fun_jac_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)

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

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

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

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[9] = {4, 2, 0, 3, 4, 0, 1, 3, 2};
static const casadi_int casadi_s3[5] = {4, 1, 0, 1, 0};
static const casadi_int casadi_s4[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s5[5] = {4, 1, 0, 1, 1};
static const casadi_int casadi_s6[5] = {4, 1, 0, 1, 2};
static const casadi_int casadi_s7[5] = {4, 1, 0, 1, 3};
static const casadi_int casadi_s8[3] = {0, 0, 0};
static const casadi_int casadi_s9[27] = {6, 4, 0, 5, 10, 15, 20, 0, 2, 3, 4, 5, 0, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 2, 3, 4, 5};
static const casadi_int casadi_s10[9] = {6, 6, 0, 0, 0, 0, 0, 0, 0};

static const casadi_real casadi_c0[16] = {1., 0., 0., 0., 0., 1., 0., 0., 2.0000000000000001e-01, 0., 1., 0., 0., 3., 0., 1.};
static const casadi_real casadi_c1[4] = {0., 1.5000000000000000e+00, 5.0000000000000000e-01, 2.0000000000000001e-01};

/* vehicle_running_acados_dyn_disc_phi_fun_jac_hess:(i0[4],i1[2],i2[4],i3[])->(o0[4],o1[6x4,20nz],o2[6x6,0nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+4, *w1=w+8, *w2=w+24, *w3=w+28, *w4=w+32, *w5=w+34, *w6=w+54, w7, w8;
  /* #0: @0 = zeros(4x1) */
  casadi_clear(w0, 4);
  /* #1: @1 = 
  [[1, 0, 0.2, 0], 
   [0, 1, 0, 3], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c0, 16, w1);
  /* #2: @2 = input[0][0] */
  casadi_copy(arg[0], 4, w2);
  /* #3: @0 = mac(@1,@2,@0) */
  for (i=0, rr=w0; i<1; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w1+j, tt=w2+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #4: @2 = zeros(4x1) */
  casadi_clear(w2, 4);
  /* #5: @3 = 
  [[-0, 00], 
   [1.5, 00], 
   [00, 0.2], 
   [0.5, 00]] */
  casadi_copy(casadi_c1, 4, w3);
  /* #6: @4 = input[1][0] */
  casadi_copy(arg[1], 2, w4);
  /* #7: @2 = mac(@3,@4,@2) */
  casadi_mtimes(w3, casadi_s2, w4, casadi_s1, w2, casadi_s0, w, 0);
  /* #8: @0 = (@0+@2) */
  for (i=0, rr=w0, cs=w2; i<4; ++i) (*rr++) += (*cs++);
  /* #9: output[0][0] = @0 */
  casadi_copy(w0, 4, res[0]);
  /* #10: @5 = zeros(6x4,20nz) */
  casadi_clear(w5, 20);
  /* #11: @0 = zeros(4x1) */
  casadi_clear(w0, 4);
  /* #12: @4 = ones(6x1,2nz) */
  casadi_fill(w4, 2, 1.);
  /* #13: {@6, NULL} = vertsplit(@4) */
  casadi_copy(w4, 2, w6);
  /* #14: @0 = mac(@3,@6,@0) */
  casadi_mtimes(w3, casadi_s2, w6, casadi_s1, w0, casadi_s0, w, 0);
  /* #15: (@5[:20:5] = @0) */
  for (rr=w5+0, ss=w0; rr!=w5+20; rr+=5) *rr = *ss++;
  /* #16: @0 = zeros(4x1) */
  casadi_clear(w0, 4);
  /* #17: @7 = ones(6x1,1nz) */
  w7 = 1.;
  /* #18: {NULL, @8} = vertsplit(@7) */
  w8 = w7;
  /* #19: @0 = mac(@1,@8,@0) */
  casadi_mtimes(w1, casadi_s4, (&w8), casadi_s3, w0, casadi_s0, w, 0);
  /* #20: (@5[1:21:5] = @0) */
  for (rr=w5+1, ss=w0; rr!=w5+21; rr+=5) *rr = *ss++;
  /* #21: @0 = zeros(4x1) */
  casadi_clear(w0, 4);
  /* #22: @8 = ones(6x1,1nz) */
  w8 = 1.;
  /* #23: {NULL, @7} = vertsplit(@8) */
  w7 = w8;
  /* #24: @0 = mac(@1,@7,@0) */
  casadi_mtimes(w1, casadi_s4, (&w7), casadi_s5, w0, casadi_s0, w, 0);
  /* #25: (@5[2:22:5] = @0) */
  for (rr=w5+2, ss=w0; rr!=w5+22; rr+=5) *rr = *ss++;
  /* #26: @0 = zeros(4x1) */
  casadi_clear(w0, 4);
  /* #27: @7 = ones(6x1,1nz) */
  w7 = 1.;
  /* #28: {NULL, @8} = vertsplit(@7) */
  w8 = w7;
  /* #29: @0 = mac(@1,@8,@0) */
  casadi_mtimes(w1, casadi_s4, (&w8), casadi_s6, w0, casadi_s0, w, 0);
  /* #30: (@5[3:23:5] = @0) */
  for (rr=w5+3, ss=w0; rr!=w5+23; rr+=5) *rr = *ss++;
  /* #31: @0 = zeros(4x1) */
  casadi_clear(w0, 4);
  /* #32: @8 = ones(6x1,1nz) */
  w8 = 1.;
  /* #33: {NULL, @7} = vertsplit(@8) */
  w7 = w8;
  /* #34: @0 = mac(@1,@7,@0) */
  casadi_mtimes(w1, casadi_s4, (&w7), casadi_s7, w0, casadi_s0, w, 0);
  /* #35: (@5[4:24:5] = @0) */
  for (rr=w5+4, ss=w0; rr!=w5+24; rr+=5) *rr = *ss++;
  /* #36: output[1][0] = @5 */
  casadi_copy(w5, 20, res[1]);
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_running_acados_dyn_disc_phi_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int vehicle_running_acados_dyn_disc_phi_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_running_acados_dyn_disc_phi_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_running_acados_dyn_disc_phi_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int vehicle_running_acados_dyn_disc_phi_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void vehicle_running_acados_dyn_disc_phi_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void vehicle_running_acados_dyn_disc_phi_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void vehicle_running_acados_dyn_disc_phi_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int vehicle_running_acados_dyn_disc_phi_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int vehicle_running_acados_dyn_disc_phi_fun_jac_hess_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real vehicle_running_acados_dyn_disc_phi_fun_jac_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_running_acados_dyn_disc_phi_fun_jac_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* vehicle_running_acados_dyn_disc_phi_fun_jac_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_running_acados_dyn_disc_phi_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    case 3: return casadi_s8;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* vehicle_running_acados_dyn_disc_phi_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s9;
    case 2: return casadi_s10;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int vehicle_running_acados_dyn_disc_phi_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 58;
  return 0;
}

CASADI_SYMBOL_EXPORT int vehicle_running_acados_dyn_disc_phi_fun_jac_hess_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 5*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 58*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
