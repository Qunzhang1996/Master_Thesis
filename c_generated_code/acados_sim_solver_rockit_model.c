/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */
// standard
#include <stdio.h>
#include <stdlib.h>

// acados
#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"
#include "acados_c/external_function_interface.h"

#include "acados/sim/sim_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/print.h"


// example specific
#include "rockit_model_model/rockit_model_model.h"
#include "acados_sim_solver_rockit_model.h"


// ** solver data **

rockit_model_sim_solver_capsule * rockit_model_acados_sim_solver_create_capsule()
{
    void* capsule_mem = malloc(sizeof(rockit_model_sim_solver_capsule));
    rockit_model_sim_solver_capsule *capsule = (rockit_model_sim_solver_capsule *) capsule_mem;

    return capsule;
}


int rockit_model_acados_sim_solver_free_capsule(rockit_model_sim_solver_capsule * capsule)
{
    free(capsule);
    return 0;
}


int rockit_model_acados_sim_create(rockit_model_sim_solver_capsule * capsule)
{
    // initialize
    const int nx = ROCKIT_MODEL_NX;
    const int nu = ROCKIT_MODEL_NU;
    const int nz = ROCKIT_MODEL_NZ;
    const int np = ROCKIT_MODEL_NP;
    bool tmp_bool;

    
    double Tsim = 0.1;

    
    // explicit ode
    capsule->sim_forw_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi));
    capsule->sim_vde_adj_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi));
    capsule->sim_expl_ode_fun_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi));

    capsule->sim_forw_vde_casadi->casadi_fun = &rockit_model_expl_vde_forw;
    capsule->sim_forw_vde_casadi->casadi_n_in = &rockit_model_expl_vde_forw_n_in;
    capsule->sim_forw_vde_casadi->casadi_n_out = &rockit_model_expl_vde_forw_n_out;
    capsule->sim_forw_vde_casadi->casadi_sparsity_in = &rockit_model_expl_vde_forw_sparsity_in;
    capsule->sim_forw_vde_casadi->casadi_sparsity_out = &rockit_model_expl_vde_forw_sparsity_out;
    capsule->sim_forw_vde_casadi->casadi_work = &rockit_model_expl_vde_forw_work;
    external_function_param_casadi_create(capsule->sim_forw_vde_casadi, np);

    capsule->sim_vde_adj_casadi->casadi_fun = &rockit_model_expl_vde_adj;
    capsule->sim_vde_adj_casadi->casadi_n_in = &rockit_model_expl_vde_adj_n_in;
    capsule->sim_vde_adj_casadi->casadi_n_out = &rockit_model_expl_vde_adj_n_out;
    capsule->sim_vde_adj_casadi->casadi_sparsity_in = &rockit_model_expl_vde_adj_sparsity_in;
    capsule->sim_vde_adj_casadi->casadi_sparsity_out = &rockit_model_expl_vde_adj_sparsity_out;
    capsule->sim_vde_adj_casadi->casadi_work = &rockit_model_expl_vde_adj_work;
    external_function_param_casadi_create(capsule->sim_vde_adj_casadi, np);

    capsule->sim_expl_ode_fun_casadi->casadi_fun = &rockit_model_expl_ode_fun;
    capsule->sim_expl_ode_fun_casadi->casadi_n_in = &rockit_model_expl_ode_fun_n_in;
    capsule->sim_expl_ode_fun_casadi->casadi_n_out = &rockit_model_expl_ode_fun_n_out;
    capsule->sim_expl_ode_fun_casadi->casadi_sparsity_in = &rockit_model_expl_ode_fun_sparsity_in;
    capsule->sim_expl_ode_fun_casadi->casadi_sparsity_out = &rockit_model_expl_ode_fun_sparsity_out;
    capsule->sim_expl_ode_fun_casadi->casadi_work = &rockit_model_expl_ode_fun_work;
    external_function_param_casadi_create(capsule->sim_expl_ode_fun_casadi, np);
    capsule->sim_expl_ode_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi));
    // external_function_param_casadi impl_dae_jac_x_xdot_u_z;
    capsule->sim_expl_ode_hess->casadi_fun = &rockit_model_expl_ode_hess;
    capsule->sim_expl_ode_hess->casadi_work = &rockit_model_expl_ode_hess_work;
    capsule->sim_expl_ode_hess->casadi_sparsity_in = &rockit_model_expl_ode_hess_sparsity_in;
    capsule->sim_expl_ode_hess->casadi_sparsity_out = &rockit_model_expl_ode_hess_sparsity_out;
    capsule->sim_expl_ode_hess->casadi_n_in = &rockit_model_expl_ode_hess_n_in;
    capsule->sim_expl_ode_hess->casadi_n_out = &rockit_model_expl_ode_hess_n_out;
    external_function_param_casadi_create(capsule->sim_expl_ode_hess, np);

    

    // sim plan & config
    sim_solver_plan_t plan;
    plan.sim_solver = ERK;

    // create correct config based on plan
    sim_config * rockit_model_sim_config = sim_config_create(plan);
    capsule->acados_sim_config = rockit_model_sim_config;

    // sim dims
    void *rockit_model_sim_dims = sim_dims_create(rockit_model_sim_config);
    capsule->acados_sim_dims = rockit_model_sim_dims;
    sim_dims_set(rockit_model_sim_config, rockit_model_sim_dims, "nx", &nx);
    sim_dims_set(rockit_model_sim_config, rockit_model_sim_dims, "nu", &nu);
    sim_dims_set(rockit_model_sim_config, rockit_model_sim_dims, "nz", &nz);


    // sim opts
    sim_opts *rockit_model_sim_opts = sim_opts_create(rockit_model_sim_config, rockit_model_sim_dims);
    capsule->acados_sim_opts = rockit_model_sim_opts;
    int tmp_int = 3;
    sim_opts_set(rockit_model_sim_config, rockit_model_sim_opts, "newton_iter", &tmp_int);
    double tmp_double = 0;
    sim_opts_set(rockit_model_sim_config, rockit_model_sim_opts, "newton_tol", &tmp_double);
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    sim_opts_set(rockit_model_sim_config, rockit_model_sim_opts, "collocation_type", &collocation_type);

 
    tmp_int = 4;
    sim_opts_set(rockit_model_sim_config, rockit_model_sim_opts, "num_stages", &tmp_int);
    tmp_int = 1;
    sim_opts_set(rockit_model_sim_config, rockit_model_sim_opts, "num_steps", &tmp_int);
    tmp_bool = 0;
    sim_opts_set(rockit_model_sim_config, rockit_model_sim_opts, "jac_reuse", &tmp_bool);


    // sim in / out
    sim_in *rockit_model_sim_in = sim_in_create(rockit_model_sim_config, rockit_model_sim_dims);
    capsule->acados_sim_in = rockit_model_sim_in;
    sim_out *rockit_model_sim_out = sim_out_create(rockit_model_sim_config, rockit_model_sim_dims);
    capsule->acados_sim_out = rockit_model_sim_out;

    sim_in_set(rockit_model_sim_config, rockit_model_sim_dims,
               rockit_model_sim_in, "T", &Tsim);

    // model functions
    rockit_model_sim_config->model_set(rockit_model_sim_in->model,
                 "expl_vde_forw", capsule->sim_forw_vde_casadi);
    rockit_model_sim_config->model_set(rockit_model_sim_in->model,
                 "expl_vde_adj", capsule->sim_vde_adj_casadi);
    rockit_model_sim_config->model_set(rockit_model_sim_in->model,
                 "expl_ode_fun", capsule->sim_expl_ode_fun_casadi);
    rockit_model_sim_config->model_set(rockit_model_sim_in->model,
                "expl_ode_hess", capsule->sim_expl_ode_hess);

    // sim solver
    sim_solver *rockit_model_sim_solver = sim_solver_create(rockit_model_sim_config,
                                               rockit_model_sim_dims, rockit_model_sim_opts);
    capsule->acados_sim_solver = rockit_model_sim_solver;


    /* initialize parameter values */
    double* p = calloc(np, sizeof(double));
    
    p[1] = 10;
    p[3] = 4.755282581475767;
    p[4] = 14.4;
    p[6] = 10;

    rockit_model_acados_sim_update_params(capsule, p, np);
    free(p);


    /* initialize input */
    // x
    double x0[4];
    for (int ii = 0; ii < 4; ii++)
        x0[ii] = 0.0;

    sim_in_set(rockit_model_sim_config, rockit_model_sim_dims,
               rockit_model_sim_in, "x", x0);


    // u
    double u0[2];
    for (int ii = 0; ii < 2; ii++)
        u0[ii] = 0.0;

    sim_in_set(rockit_model_sim_config, rockit_model_sim_dims,
               rockit_model_sim_in, "u", u0);

    // S_forw
    double S_forw[24];
    for (int ii = 0; ii < 24; ii++)
        S_forw[ii] = 0.0;
    for (int ii = 0; ii < 4; ii++)
        S_forw[ii + ii * 4 ] = 1.0;


    sim_in_set(rockit_model_sim_config, rockit_model_sim_dims,
               rockit_model_sim_in, "S_forw", S_forw);

    int status = sim_precompute(rockit_model_sim_solver, rockit_model_sim_in, rockit_model_sim_out);

    return status;
}


int rockit_model_acados_sim_solve(rockit_model_sim_solver_capsule *capsule)
{
    // integrate dynamics using acados sim_solver
    int status = sim_solve(capsule->acados_sim_solver,
                           capsule->acados_sim_in, capsule->acados_sim_out);
    if (status != 0)
        printf("error in rockit_model_acados_sim_solve()! Exiting.\n");

    return status;
}


int rockit_model_acados_sim_free(rockit_model_sim_solver_capsule *capsule)
{
    // free memory
    sim_solver_destroy(capsule->acados_sim_solver);
    sim_in_destroy(capsule->acados_sim_in);
    sim_out_destroy(capsule->acados_sim_out);
    sim_opts_destroy(capsule->acados_sim_opts);
    sim_dims_destroy(capsule->acados_sim_dims);
    sim_config_destroy(capsule->acados_sim_config);

    // free external function
    external_function_param_casadi_free(capsule->sim_forw_vde_casadi);
    external_function_param_casadi_free(capsule->sim_vde_adj_casadi);
    external_function_param_casadi_free(capsule->sim_expl_ode_fun_casadi);
    free(capsule->sim_forw_vde_casadi);
    free(capsule->sim_vde_adj_casadi);
    free(capsule->sim_expl_ode_fun_casadi);
    external_function_param_casadi_free(capsule->sim_expl_ode_hess);
    free(capsule->sim_expl_ode_hess);

    return 0;
}


int rockit_model_acados_sim_update_params(rockit_model_sim_solver_capsule *capsule, double *p, int np)
{
    int status = 0;
    int casadi_np = ROCKIT_MODEL_NP;

    if (casadi_np != np) {
        printf("rockit_model_acados_sim_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    capsule->sim_forw_vde_casadi[0].set_param(capsule->sim_forw_vde_casadi, p);
    capsule->sim_vde_adj_casadi[0].set_param(capsule->sim_vde_adj_casadi, p);
    capsule->sim_expl_ode_fun_casadi[0].set_param(capsule->sim_expl_ode_fun_casadi, p);
    capsule->sim_expl_ode_hess[0].set_param(capsule->sim_expl_ode_hess, p);

    return status;
}

/* getters pointers to C objects*/
sim_config * rockit_model_acados_get_sim_config(rockit_model_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_config;
};

sim_in * rockit_model_acados_get_sim_in(rockit_model_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_in;
};

sim_out * rockit_model_acados_get_sim_out(rockit_model_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_out;
};

void * rockit_model_acados_get_sim_dims(rockit_model_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_dims;
};

sim_opts * rockit_model_acados_get_sim_opts(rockit_model_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_opts;
};

sim_solver  * rockit_model_acados_get_sim_solver(rockit_model_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_solver;
};

