#!/usr/bin/env python
# coding=UTF-8
'''
Author: Wei Luo
Date: 2021-03-15 22:43:48
LastEditors: Wei Luo
LastEditTime: 2021-03-17 23:28:55
Note: Note
'''

import numpy as np
from casadi import*
from acados_template import AcadosModel

class MobileRobotModel(object):
    def __init__(self,):
        model = AcadosModel() #  ca.types.SimpleNamespace()
        constraint = types.SimpleNamespace()
        # control inputs
        v = SX.sym('v')
        omega = SX.sym('omega')
        controls = vertcat(v, omega)
        # n_controls = controls.size()[0]
        # model states
        x = SX.sym('x')
        y = SX.sym('y')
        theta = SX.sym('theta')
        states = vertcat(x, y, theta)

        rhs = [v*cos(theta), v*sin(theta), omega]

        # function
        f = Function('f', [states, controls], [vcat(rhs)], ['state', 'control_input'], ['rhs'])
        # f_expl = ca.vcat(rhs)
        # acados model
        x_dot = SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'mobile_robot'

        # constraint
        constraint.v_max = 0.6
        constraint.v_min = -0.6
        constraint.omega_max = np.pi/4.0
        constraint.omega_min = -np.pi/4.0
        constraint.x_min = -2.
        constraint.x_max = 2.
        constraint.y_min = -2.
        constraint.y_max = 2.
        constraint.expr = vcat([v, omega])

        self.model = model
        self.constraint = constraint