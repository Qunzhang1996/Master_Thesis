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
  #define CASADI_PREFIX(ID) Truck_bicycle_gnsf_get_matrices_fun_ ## ID
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
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)

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

static const casadi_int casadi_s0[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s1[9] = {2, 2, 0, 2, 4, 0, 1, 0, 1};
static const casadi_int casadi_s2[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s3[7] = {1, 2, 0, 1, 2, 0, 0};
static const casadi_int casadi_s4[3] = {1, 0, 0};
static const casadi_int casadi_s5[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s6[4] = {0, 1, 0, 0};

/* Truck_bicycle_gnsf_get_matrices_fun:(i0)->(o0[2x2],o1[2x2],o2[2],o3[2x2],o4[1x2],o5[1x2],o6[1x0],o7[1x2],o8[2x2],o9[2],o10[2x2],o11[2x2],o12,o13,o14[4],o15[0],o16[2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  if (res[0]!=0) res[0][2]=a0;
  if (res[0]!=0) res[0][3]=a0;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  a1=-1.;
  if (res[1]!=0) res[1][2]=a1;
  if (res[1]!=0) res[1][3]=a0;
  if (res[2]!=0) res[2][0]=a0;
  a2=1.;
  if (res[2]!=0) res[2][1]=a2;
  if (res[3]!=0) res[3][0]=a1;
  if (res[3]!=0) res[3][1]=a0;
  if (res[3]!=0) res[3][2]=a0;
  if (res[3]!=0) res[3][3]=a1;
  if (res[4]!=0) res[4][0]=a2;
  if (res[4]!=0) res[4][1]=a0;
  if (res[5]!=0) res[5][0]=a0;
  if (res[5]!=0) res[5][1]=a0;
  if (res[7]!=0) res[7][0]=a2;
  if (res[7]!=0) res[7][1]=a0;
  if (res[8]!=0) res[8][0]=a0;
  if (res[8]!=0) res[8][1]=a0;
  if (res[8]!=0) res[8][2]=a0;
  if (res[8]!=0) res[8][3]=a0;
  if (res[9]!=0) res[9][0]=a0;
  if (res[9]!=0) res[9][1]=a0;
  if (res[10]!=0) res[10][0]=a1;
  if (res[10]!=0) res[10][1]=a0;
  if (res[10]!=0) res[10][2]=a0;
  if (res[10]!=0) res[10][3]=a1;
  if (res[11]!=0) res[11][0]=a0;
  if (res[11]!=0) res[11][1]=a0;
  if (res[11]!=0) res[11][2]=a0;
  if (res[11]!=0) res[11][3]=a0;
  if (res[12]!=0) res[12][0]=a2;
  if (res[13]!=0) res[13][0]=a0;
  a2=2.;
  if (res[14]!=0) res[14][0]=a2;
  a1=3.;
  if (res[14]!=0) res[14][1]=a1;
  if (res[14]!=0) res[14][2]=a2;
  if (res[14]!=0) res[14][3]=a1;
  if (res[16]!=0) res[16][0]=a0;
  if (res[16]!=0) res[16][1]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_get_matrices_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_get_matrices_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_get_matrices_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_get_matrices_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_get_matrices_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_get_matrices_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_get_matrices_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Truck_bicycle_gnsf_get_matrices_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Truck_bicycle_gnsf_get_matrices_fun_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int Truck_bicycle_gnsf_get_matrices_fun_n_out(void) { return 17;}

CASADI_SYMBOL_EXPORT casadi_real Truck_bicycle_gnsf_get_matrices_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Truck_bicycle_gnsf_get_matrices_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Truck_bicycle_gnsf_get_matrices_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    case 5: return "o5";
    case 6: return "o6";
    case 7: return "o7";
    case 8: return "o8";
    case 9: return "o9";
    case 10: return "o10";
    case 11: return "o11";
    case 12: return "o12";
    case 13: return "o13";
    case 14: return "o14";
    case 15: return "o15";
    case 16: return "o16";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Truck_bicycle_gnsf_get_matrices_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Truck_bicycle_gnsf_get_matrices_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s1;
    case 4: return casadi_s3;
    case 5: return casadi_s3;
    case 6: return casadi_s4;
    case 7: return casadi_s3;
    case 8: return casadi_s1;
    case 9: return casadi_s2;
    case 10: return casadi_s1;
    case 11: return casadi_s1;
    case 12: return casadi_s0;
    case 13: return casadi_s0;
    case 14: return casadi_s5;
    case 15: return casadi_s6;
    case 16: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_get_matrices_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 17;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Truck_bicycle_gnsf_get_matrices_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 17*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
