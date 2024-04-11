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
  #define CASADI_PREFIX(ID) Truck_bicycle_gnsf_phi_jac_y_uhat_ ## ID
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

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s1[3] = {0, 0, 0};

/* Truck_bicycle_gnsf_phi_jac_y_uhat:(i0,i1,i2[])->(o0,o1) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1;
  a0=arg[1]? arg[1][0] : 0;
  a1=tan(a0);
  a1=(-a1);
  if (res[0]!=0) res[0][0]=a1;
  a1=arg[0]? arg[0][0] : 0;
  a0=cos(a0);
  a0=casadi_sq(a0);
  a1=(a1/a0);
  a1=(-a1);
  if (res[1]!=0) res[1][0]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_phi_jac_y_uhat(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_phi_jac_y_uhat_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_phi_jac_y_uhat_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_phi_jac_y_uhat_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_phi_jac_y_uhat_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_phi_jac_y_uhat_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_phi_jac_y_uhat_incref(void) {
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_phi_jac_y_uhat_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Truck_bicycle_gnsf_phi_jac_y_uhat_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int Truck_bicycle_gnsf_phi_jac_y_uhat_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real Truck_bicycle_gnsf_phi_jac_y_uhat_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Truck_bicycle_gnsf_phi_jac_y_uhat_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Truck_bicycle_gnsf_phi_jac_y_uhat_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Truck_bicycle_gnsf_phi_jac_y_uhat_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Truck_bicycle_gnsf_phi_jac_y_uhat_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_phi_jac_y_uhat_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_phi_jac_y_uhat_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
