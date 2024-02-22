from casadi import *
import numpy as np
from matplotlib import pyplot as plt
from helpers import *

class makeController:
    """
        Creates a MPC based on current vehicle, traffic and scenario
    """
    def __init__(self, vehicle,traffic,scenario,N,opts,dt):
        self.vehicle = vehicle
        self.traffic = traffic
        self.scenario = scenario
        self.opts = opts

        # Get constants
        self.Nveh = self.traffic.getDim()
        self.N = N
        self.vehWidth,_,_,_ = self.vehicle.getSize()
        self.nx,self.nu,self.nrefx,self.nrefu = self.vehicle.getSystemDim()
        self.roadMin, self.roadMax, self.laneCenters = self.scenario.getRoad()
        self.opts = opts

        # Setup vehicle integration scheme
        self.vehicle.integrator(opts["integrator"],dt)
        self.F_x  = self.vehicle.getIntegrator()
        self.scaling = self.vehicle.getScaling()

        # Create Opti Stack
        self.opti = Opti()

        # # Initialize opti stack
        self.x = self.opti.variable(self.nx,self.N+1)
        self.u = self.opti.variable(self.nu,self.N)
        self.refx = self.opti.parameter(self.nrefx,self.N+1)
        self.refu = self.opti.parameter(self.nrefu,self.N)
        self.x0 = self.opti.parameter(self.nx,1)

        if opts["version"] == "trailing":
            self.lead = self.opti.parameter(1,self.N+1)
        else:
            self.lead = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_x = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_y = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_sign = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_shift = self.opti.parameter(self.Nveh,self.N+1)
            self.traffic_flip = self.opti.parameter(self.Nveh,self.N+1)

        # # Solver
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)
        self.opti.solver(self.opts["solver"], p_opts,s_opts)

    def setStateEqconstraints(self):
        # # Set equality state Constraints
        for i in range(self.N):
            self.opti.subject_to(self.x[:,i+1] == self.F_x(self.x[:,i],self.u[:,i]))

        self.opti.subject_to(self.x[:,0] == self.x0)

    def setInEqConstraints(self):
        self.opti.subject_to(self.opti.bounded(self.roadMin+self.vehWidth/2,self.x[1,:],self.roadMax-self.vehWidth/2))

        lbu,ubu = self.vehicle.uConstraints()
        lbx,ubx = self.vehicle.xConstraints()
        self.opti.subject_to(self.opti.bounded(lbu,self.u,ubu))
        self.opti.subject_to(self.opti.bounded(lbx,self.x,ubx))

        self.opti.subject_to(self.opti.bounded(0,self.x[2,:],self.scenario.vmax))
    
    def setTrafficConstraints(self):
        self.S = self.scenario.constraint(self.traffic,self.opts)

        if self.scenario.name == 'simpleOvertake':
            for i in range(self.Nveh):
                self.opti.subject_to(self.traffic_flip[i,:] * self.x[1,:] 
                                    >= self.traffic_flip[i,:] * self.S[i](self.x[0,:], self.traffic_x[i,:], self.traffic_y[i,:],
                                        self.traffic_sign[i,:], self.traffic_shift[i,:]))

        elif self.scenario.name == 'trailing':
            self.scenario.setEgoLane()
            self.scenario.getLeadVehicle(self.traffic)
            self.opti.subject_to(self.x[0,:] <= self.S(self.lead))

    def setCost(self):
        L,Lf = self.vehicle.getCost()
        self.opti.minimize(getTotalCost(L,Lf,self.x,self.u,self.refx,self.refu,self.N))

    def setController(self):
        """
        Sets all constraints and cost
        """
        # Constraints
        self.setStateEqconstraints()
        self.setInEqConstraints()
        self.setTrafficConstraints()

        # Cost
        self.setCost()

    def getFunction(self):
        if self.opts["version"] == "trailing":
            return self.opti.to_function('MPC',[self.x0,self.refx,self.refu,self.lead],[self.x,self.u],
                    ['x0','refx','refu','lead'],['x_opt','u_opt'])
        else:
            return self.opti.to_function('MPC',[self.x0,self.refx,self.refu,
                    self.traffic_x,self.traffic_y,self.traffic_sign,self.traffic_shift,self.traffic_flip],[self.x,self.u],
                    ['x0','refx','refu','t_x','t_y','t_sign','t_shift','t_flip'],['x_opt','u_opt'])

    def getVersion(self):
        return self.opts["version"]

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

    def testSolver(self,traffic):
        """
        Most errors are not visible when using the "to_function" approach
        This functions runs one instance of the MPC controller and plots scenario at failiure
        """
        Nveh = traffic.Nveh
        # # Initialize 
        x_iter = DM(int(self.nx),1)

        if self.vehicle.name == "truck_trailer_bicycle":
            x_iter[:] = [0,self.laneCenters[0],30/3.6,0,0]
            refx_in = [0,self.laneCenters[0],60/3.6,0,0]
            refu_in = [0,0]
        elif self.vehicle.name == "truck_trailer_bicycle_energyEff": 
            x_iter[:] = [0,self.laneCenters[0],30/3.6,0,0]
            refx_in = [0,self.laneCenters[0],60/3.6,0,0]
            refu_in = [0,0,0]
        else:
            x_iter[:] = [0,self.laneCenters[0],30/3.6,0]

            refx_in = [0,self.laneCenters[0],60/3.6,0]
            refu_in = [0,0]

        refx_out,refu_out = self.scenario.getReference(refx_in,refu_in)

        traffic_x = DM(self.N+1,self.Nveh)
        traffic_y = DM(self.N+1,self.Nveh)
        traffic_sign = DM(self.N+1,self.Nveh)
        traffic_shift = DM(self.N+1,self.Nveh)
        traffic_flip = DM(self.N+1,self.Nveh)

        traffic_state = np.zeros((5,self.N+1,Nveh))                  # ! Should be changed to DM in final implementation!
        traffic_state[:2,:,:] = traffic.prediction()[:2,:,:]
        traffic_state[2,:,:],traffic_state[3,:,:],traffic_state[4,:,:] = self.testControllerParameters(self.opts["version"],traffic_state)

        # for i in range(self.Nveh):
            # x_lead[i,:] = self.traffic.prediction()[0,:,i]


        if self.scenario.name == 'trailing':
            idx = self.scenario.getLeadVehicle(self.traffic)                             # Also sets egoLANE!
            if len(idx) == 0:         # Move barrier very far forward
                x_traffic = DM.ones(1,self.N+1)*10000
            else:
                # x_traffic = x_lead[idx[0],:]
                pass
        else:
            # x_traffic = x_lead
            traffic_x = traffic_state[0,:,:].T
            traffic_y = traffic_state[1,:,:].T
            traffic_sign = traffic_state[2,:,:].T
            traffic_shift = traffic_state[3,:,:].T
            traffic_flip = traffic_state[4,:,:].T

            self.opti.set_value(self.refx,refx_out)
            self.opti.set_value(self.refu,refu_out)
            self.opti.set_value(self.traffic_x,traffic_x)
            self.opti.set_value(self.traffic_y,traffic_y)
            self.opti.set_value(self.traffic_sign,traffic_sign)
            self.opti.set_value(self.traffic_shift,traffic_shift)
            self.opti.set_value(self.traffic_flip,traffic_flip)
            self.opti.set_value(self.x0,x_iter)

        try:
            sol = self.opti.solve()
            print("-----------")
            print("Test successful")
            print("Controller version:",self.opts["version"])
            print("-----------")
            # raise ValueError
        except:
            x_fail = np.array(self.opti.debug.value(self.x))
            print(self.opti.debug.value(self.x))
            print(self.opti.debug.value(self.u))
            print(x_fail[0,:])

            plt.plot(x_fail[0,:],x_fail[1,:])

            X_traffic = np.zeros((1,200))
            X_traffic[0] = 50

            ConstraintTEST = np.zeros((Nveh,200))
            for j in range(Nveh):
                for i in range(200):
                    x_test = np.arange(i,i+self.N+1)
                    
                    t_x_i = traffic_state[0,:,:]
                    t_y_i = traffic_state[1,:,:]
                    t_sign_i = traffic_state[2,:,:]
                    t_shift_i = traffic_state[3,:,:]
                    t_flip_i = traffic_state[4,:,:]
                    # print(np.shape(t_shift_i))
                    ConstraintTEST[j,i] = self.S[j](x_test,t_x_i[:,j],t_y_i[:,j],t_sign_i[:,j],t_shift_i[:,j])[0]
                    # plt.scatter(x_lead[0,0],x_lead[0,1],marker = "*")

                    # Update leadVeh
                    # traffic.update()
                    # X_traffic[:,i] = self.traffic.vehicles[0].getState()[0]
                plt.plot(np.arange(0,200),ConstraintTEST[j,:])
                plt.plot(np.arange(0,200),np.ones((1,200))[0]*13/2,'--',color = 'r')
                plt.plot(np.arange(0,200),np.zeros((1,200))[0],'--',color = 'r')
                plt.plot(np.arange(0,200),np.ones((1,200))[0]*-6.5,'-',color = 'k')
                plt.plot(np.arange(0,200),np.ones((1,200))[0]*13,'-',color = 'k')

                x_lead_j= self.traffic.prediction()[:,0,j]
                plt.scatter(x_lead_j[0], x_lead_j[1],marker = "*")
                
            plt.show()

            raise ValueError("Simulation initilization failed, check traffic initialization")
        
    def testControllerParameters(self,version,xy):
        sign = np.zeros((self.Nveh,self.N+1))
        shift = np.zeros((self.Nveh,self.N+1))
        flip = np.ones((self.Nveh,self.N+1))

        if version == "leftChange":
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if xy[1,jj,ii] > 6.5:
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * 6.5
                        flip[ii,jj] = -1
                    elif xy[1,jj,ii] < 0:
                        sign[ii,jj] = 1
                        shift[ii,jj] = -6.5
                    else:
                        sign[ii,jj] = 1
                        shift[ii,jj] = 0

        elif version == "rightChange":
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if xy[1,jj,ii] > 6.5:
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * 6.5
                        flip[ii,jj] = -1
                    elif xy[1,jj,ii] < 0:
                        sign[ii,jj] = 1
                        shift[ii,jj] = -6.5
                    else:
                        sign[ii,jj] = -1
                        shift[ii,jj] = 6.5
                        flip[ii,jj] = -1

        return sign.T, shift.T, flip.T


class makeDecisionMaster:
    def __init__(self,vehicle,traffic,controllers,scenarios,rl_agent,changeHorizon = 10,forgettingFact = 0.90):
        self.vehicle = vehicle
        self.traffic = traffic
        self.controllers = controllers
        self.scenarios = scenarios
        self.rl_agent = rl_agent

        self.laneCenters = self.scenarios[1].getRoad()[2]
        self.laneWidth = self.laneCenters[0] * 2

        self.egoLane = self.scenarios[0].getEgoLane()
        self.egoPx = self.vehicle.x_init[0]

        self.nx,self.nu,self.nrefx,self.nrefu = vehicle.getSystemDim()
        self.N = vehicle.N
        self.Nveh = self.traffic.getDim()

        self.doRouteGoalScenario = 0

        self.state = np.zeros((self.nx,))
        self.x_pred = np.zeros((self.nx,self.N))
        self.u_pred = np.zeros((self.nu,self.N))
        self.tol = 0.1
        self.consecutiveErrors = 0
        self.changeHorizon = changeHorizon
        self.forgettingFact = forgettingFact

        self.L, self.Lf = vehicle.getCost()
        self.MPCs = []
        for i in range(len(self.controllers)):
            self.MPCs.append(controllers[i].getFunction())

        self.errors = 0

        self.decisionLog = []

    def checkSolution(self,x_pred_new,u_pred_new):
        """
        Checks if the MPC returned a strange solution
         - If that is the case, fall back to the previous solution
         - The number of times this occurs is presented as "Error count"
        """
        cond1 = (x_pred_new[0,0]) < (self.x_pred[0,0] - self.tol)
        cond2 = ((x_pred_new[1,0] - x_pred_new[1,1]) > 1)
        if  (cond1 or cond2) and (self.consecutiveErrors < self.N-1):
            self.consecutiveErrors += 1
            self.errors += 1
            return self.x_pred[:,self.consecutiveErrors], self.u_pred[:,self.consecutiveErrors], self.x_pred
        else:
            self.consecutiveErrors = 0
            self.x_pred = x_pred_new
            self.u_pred = u_pred_new
            return x_pred_new[:,0], u_pred_new[:,0], self.x_pred

    def storeInput(self,input):
        """
        Stores the current states sent from main file
        """
        self.x_iter, self.refxL_out, self.refxR_out, self.refxT_out, self.refu_out, self.x_lead,self.traffic_state = input

    def setDecisionCost(self,q):
        """
        Sets costs of changing a decisions
        """
        self.decisionQ = q


    def setRouteGoal(self,lane = -1,distance = 1000, cost = 100,xmax = 2000):
        # Lane: [left = 1, middle = 0, right == !!(2 or -1))!!  ]
        # Initalize higway goal (e.g. exit)
        self.goalP_x = distance
        self.goalLane = lane
        self.goalD_exp = 0.4
        self.goalD_xmax = xmax
        self.goalRef = np.ones((1,self.N+1)) * self.laneCenters[lane]
        self.doRouteGoalScenario = 1
        self.goalCost = cost
        self.goalAccomplished = 0

    def costRouteGoal(self,i):
        # i == 0 -> left change, i == 1 -> right change, i == 2 -> trail
        if self.doRouteGoalScenario and (self.goalP_x - self.egoPx < self.goalD_xmax):
            # Check if goal is reached
            if self.goalP_x - self.egoPx < 0:
                self.goalD_xmax = -1e5             # Deactivates the goal cost
                if self.egoLane == self.goalLane:
                    self.goalAccomplished = 1

            currentLane = self.egoLane
            # Find the best action in terms of reaching the goal
            if self.goalLane == currentLane:
                # If the goal lane is the current lane, dont change lane
                bestChoice = 2          
            elif currentLane == 1:
                # We are left and the goal is not left, change
                bestChoice = 0
            elif currentLane == -1:
                # We are right and the goal is not right, change
                bestChoice = 1
            else:
                # We are center and the gol is not center, change
                if self.goalLane == 1:
                    # If goal is left change left
                    bestChoice = 0
                elif self.goalLane == -1:
                    # If the goal is right change right
                    bestChoice = 1
            cost = (1 - ((self.goalP_x-self.egoPx) / self.goalD_xmax )** 0.4 ) * np.minimum(np.abs(i-bestChoice),1)
            return self.goalCost * cost
        else:
            # We dont consider any goal
            return 0

    def costDecision(self,decision):
        """
        Returns cost of the current decision based on the past (i == changeHorizon) decisions
        """
        cost = 0
        for i in range(self.changeHorizon):
            cost += self.decisionQ * (self.forgettingFact ** i) * (decision - self.decisionLog[i]) ** 2
        return cost

    def getDecision(self,costs):
        """
        Find optimal choice out of the three controllers
        """
        costMPC = np.array(costs)
        costTotal = np.zeros((3,))

        for i in range(3):
            costTotal[i] = self.costDecision(i) + costMPC[i] + self.costRouteGoal(i)
        return np.argmin(costTotal)

    def updateReference(self):
        """
        Updates the y position reference for each controller based on the current lane
        """

        py_ego = self.vehicle.getPosition()[1]
        self.egoPx = self.vehicle.getPosition()[0]
        refu_in = [0,0,0]                                     # To work with function reference (update?)

        refxT_in,refxL_in,refxR_in = self.vehicle.getReferences()

        tol = 0.2
        if py_ego >= self.laneCenters[1]:
            # Set left reference to mid lane
            # Set trailing reference to left lane
            refxT_in[1] = self.laneCenters[1]
            refxL_in[1] = self.laneCenters[0]
            refxR_in[1] = self.laneCenters[2]

        elif py_ego < self.laneCenters[2]:
            # Set right reference to right lane
            # Set trailing reference to right lane
            refxT_in[1] = self.laneCenters[2]
            refxL_in[1] = self.laneCenters[1]
            refxR_in[1] = self.laneCenters[0]

        elif abs(py_ego - self.laneCenters[0]) < tol:
            # Set left reference to left lane
            # Set right reference to right lane
            # Set trailing reference to middle lane
            # refxT_in[1] = self.laneCenters[0]
            refxL_in[1] = self.laneCenters[1]
            refxR_in[1] = self.laneCenters[2]

        # Trailing reference should always be the current Lane!
        refxT_in[1] = self.laneCenters[self.egoLane]
        
        self.refxT,_ = self.scenarios[1].getReference(refxT_in,refu_in)
        self.refxL,_ = self.scenarios[1].getReference(refxL_in,refu_in)
        self.refxR,_ = self.scenarios[1].getReference(refxR_in,refu_in)
        
        return self.refxL,self.refxR,self.refxT

    def removeDeviation(self):
        """
        Centers the x-position around 0 (to fix nummerical issues)
        """
        # Store current values of changes
        self.egoPx = self.x_iter[0]
        # Alter initialization of MPC
        # # X-position
        self.x_iter[0] = 0
        for i in range(self.Nveh):
            self.x_lead[i,:] = self.x_lead[i,:] - self.egoPx
        
        self.traffic_state[0,:,:] = self.traffic_state[0,:,:] - self.egoPx
        self.traffic_state[0,:,:] = np.clip(self.traffic_state[0,:,:],-600,600)

    def returnDeviation(self,X,U):
        """
        # Adds back the deviations that where removed in the above function
        """
        self.x_iter[0] = self.egoPx
        X[0,:] = X[0,:] + self.egoPx
        return X, U

    def setControllerParameters(self,version):
        """
        Sets traffic parameters, to be used in the MPC controllers
        """

        sign = np.zeros((self.Nveh,self.N+1))
        shift = np.zeros((self.Nveh,self.N+1))
        flip = np.ones((self.Nveh,self.N+1))

        if version == "leftChange":
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if self.traffic_state[1,jj,ii] > self.laneWidth:
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * self.laneWidth
                        flip[ii,jj] = -1
                    elif self.traffic_state[1,jj,ii] < 0:
                        sign[ii,jj] = 1
                        shift[ii,jj] = -self.laneWidth
                    else:
                        sign[ii,jj] = 1
                        shift[ii,jj] = 0

        elif version == "rightChange":
            for ii in range(self.Nveh):
                for jj in range(self.N+1):
                    if self.traffic_state[1,jj,ii] > self.laneWidth:
                        sign[ii,jj] = -1
                        shift[ii,jj] = 2 * self.laneWidth
                        flip[ii,jj] = -1
                    elif self.traffic_state[1,jj,ii] < 0:
                        sign[ii,jj] = 1
                        shift[ii,jj] = -self.laneWidth
                    else:
                        sign[ii,jj] = -1
                        shift[ii,jj] = self.laneWidth
                        flip[ii,jj] = -1

        self.traffic_state[2,:,:] = sign.T
        self.traffic_state[3,:,:] = shift.T
        self.traffic_state[4,:,:] = flip.T

    def chooseController(self):
        """
        Main function, finds optimal choice of controller for the current step
        """
        self.egoLane = self.scenarios[0].getEgoLane()

        # Revoke controller usage if initialized unfeasible
        self.doTrailing = 1
        if self.egoLane == 0:
            self.doLeft = 1
            self.doRight = 1
        elif self.egoLane == 1:
            self.doLeft = 1
            self.doRight = 0
        elif self.egoLane == -1:
            self.doLeft = 0
            self.doRight = 1

        # Set controller decision based on RL agent
        self.RL_decision = self.rl_agent.getDecision()

        if not(np.isnan(self.RL_decision)):
            if self.RL_decision == 0:
                self.doTrailing = 0
                self.doLeft = 1
                self.doRight = 0
            elif self.RL_decision == 1:
                self.doTrailing = 0
                self.doLeft = 0
                self.doRight = 1
            elif self.RL_decision == 2:
                self.doTrailing = 1
                self.doLeft = 0
                self.doRight = 0

        # Initialize costs as very large number
        costT = DM([1e10])
        costR = DM([1e10])
        costL = DM([1e10])

        # Set to deviation variables
        self.removeDeviation()

        # Calculate cost of different solutions
        if self.doTrailing:
            # Trailing solver
            idx = self.scenarios[0].getLeadVehicle(self.traffic)                             # Also sets egoLANE!
            if len(idx) == 0:         # Move barrier very far forward
                x_traffic = DM(1,self.N+1)
                x_traffic[0,:] = self.x_iter[0] + 200
            else:
                x_traffic = self.x_lead[idx[0],:]
            
            x_testT,u_testT = self.MPCs[2](self.x_iter,self.refxT_out,self.refu_out,x_traffic)

            costT = getTotalCost(self.L,self.Lf,x_testT,u_testT,self.refxT_out,self.refu_out,self.N)

        if self.doLeft:
            # Changing between left and middle lane
            self.setControllerParameters(self.controllers[0].opts["version"])
            x_testL,u_testL = self.MPCs[0](self.x_iter, self.refxL_out, self.refu_out,
                             self.traffic_state[0,:,:].T,self.traffic_state[1,:,:].T,self.traffic_state[2,:,:].T,
                             self.traffic_state[3,:,:].T,self.traffic_state[4,:,:].T)

            costL = getTotalCost(self.L,self.Lf,x_testL,u_testL,self.refxL_out,self.refu_out,self.N)

        if self.doRight:
            # Changing between right and middle lane
            self.setControllerParameters(self.controllers[1].opts["version"])
            x_testR,u_testR = self.MPCs[1](self.x_iter, self.refxR_out, self.refu_out,
                             self.traffic_state[0,:,:].T,self.traffic_state[1,:,:].T,self.traffic_state[2,:,:].T,
                             self.traffic_state[3,:,:].T,self.traffic_state[4,:,:].T)

            costR = getTotalCost(self.L,self.Lf,x_testR,u_testR,self.refxR_out,self.refu_out,self.N)

        if len(self.decisionLog) >= self.changeHorizon:
            decision_i = self.getDecision([costL,costR,costT])
            self.decisionLog.insert(0,decision_i)
            self.decisionLog.pop()
        else:
            decision_i = np.argmin(np.array([costL,costR,costT]))
            self.decisionLog.insert(0,decision_i)
        
        print('Decision: ',self.controllers[decision_i].opts["version"])
        
        if decision_i == 0:
            X = x_testL
            U = u_testL
        elif decision_i == 1:
            X = x_testR
            U = u_testR
        else:
            X = x_testT
            U = u_testT

        X, U = self.returnDeviation(X,U)

        x_ok, u_ok, X = self.checkSolution(X,U)

        return x_ok,u_ok, X

    def getErrorCount(self):
        """
        Returns the amount of strange solutions encountered
        """
        return self.errors

    def getGoalStatus(self):
        if self.doRouteGoalScenario == 1:
            if self.goalAccomplished == 1:
                return "Succesfully reached"
            else:
                return "Not reached"
        else:
            return "Was not considered"