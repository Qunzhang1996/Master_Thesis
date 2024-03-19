# here is the pure pursuit algorithm
alpha = np.arcsin(x_iter - ego_y)
steer = arctan(2*wb*sin(alpha)/vx_ref)