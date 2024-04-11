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
  #define CASADI_PREFIX(ID) Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

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

static const casadi_int casadi_s0[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s1[4] = {0, 1, 0, 0};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[13] = {2, 6, 0, 2, 4, 4, 4, 4, 4, 0, 1, 0, 1};

/* Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz:(i0[2],i1[2],i2[0],i3[2],i4[])->(o0[2],o1[2x6,4nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4;
  a0=arg[0]? arg[0][0] : 0;
  a1=arg[0]? arg[0][1] : 0;
  a2=cos(a1);
  a3=(a0*a2);
  a3=(-a3);
  if (res[0]!=0) res[0][0]=a3;
  a3=sin(a1);
  a4=(a0*a3);
  a4=(-a4);
  if (res[0]!=0) res[0][1]=a4;
  a2=(-a2);
  if (res[1]!=0) res[1][0]=a2;
  a3=(-a3);
  if (res[1]!=0) res[1][1]=a3;
  a3=sin(a1);
  a3=(a0*a3);
  if (res[1]!=0) res[1][2]=a3;
  a1=cos(a1);
  a0=(a0*a1);
  a0=(-a0);
  if (res[1]!=0) res[1][3]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_incref(void) {
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s0;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_f_lo_fun_jac_x1k1uz_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
