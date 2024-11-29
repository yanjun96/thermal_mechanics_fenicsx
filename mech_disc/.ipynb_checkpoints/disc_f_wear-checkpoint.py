################################################################
from disc_f import *


def solve_heat(Ti, u, num_steps, dt, x_co, y_co, angular_r, \
               t_brake, domain, S_rub_circle, fdim,\
               rho, c, v, radiation, k, h, f, Tm, g,\
               ds, xdmf, b_con, bc, plotter, warped,\
               mesh_name1, mesh_brake, pad_v_tag, z4,\
               z1, x_co_zone, u_n, alpha_thermal, penalty_param, P ):
    import numpy as np
  
    T_array = [(0, [Ti for _ in range(len(u.x.array))])]
    total_degree = 0
    t = 0
    fraction_c = []
    for i in range(num_steps):
        
         t += dt[i]
        
         if i == 0: 
            u.x.array[:] = np.full(len(u.x.array), Ti)       
        
         if i == 0:
            x_co_new = x_co
            y_co_new = y_co
         else:
            pass  

         if i == 0:
            friction_c1 = 1
            d_wear = []
             

         K = 6.7e-6                    # no dimension
         W = P[i] * 200/10000          # 2.74 mpa/ 200 cm2  
         d1 = angular_r * 251.5/ 1000  # m
         H = (2.8e8) / 3               # N/m2, yield strength / 3
         V_wear = K * W * d1 / H       # wear volume, m3
         d_wear0 = V_wear*(1e9) / (200*100 /friction_c1) 
         d_wear.append(d_wear0)
   
         total_degree += angular_r  # Incrementing degree in each step  
       
         if i == 0:
             u_d0, Vu, aM, LM, bcu, u_, domain_pad = \
             T_S_deformation_solve (mesh_name1, u, \
                                    mesh_brake, pad_v_tag, z4, u, alpha_thermal)
         else:
             u_d0, Vu, aM, LM, bcu, u_, domain_pad = \
             T_S_deformation_solve (mesh_name1, u,\
                                    mesh_brake, pad_v_tag, z4, u_pre_solve, alpha_thermal)
         # u_d1 is the new deformation with contact force, mm.
         u_d1 = penalty_method_contact(z1, Vu, u_d0, aM, LM, u_, bcu, penalty_param )
             
         # calculate new contact coordinates and contact incicies of u_d, deformation.
         deformed_co, new_c   = get_new_contact_nodes(x_co_zone, domain_pad, u_d1, Vu, z1, \
                                                      x_co, y_co, S_rub_circle, i  )
         # find new contact coordinates and rub radius.
         x_co_new1, y_co_new1, r_rub_new, S_total_new, S_rub_circle_new = get_r_xco_yco (deformed_co, new_c )
         S_rub_circle = S_rub_circle_new
         x_co_new, y_co_new = rub_rotation(x_co_new1, y_co_new1, total_degree)
        
         r_square = r_rub_new**2
         fraction_c1 = S_total_new /200
         fraction_c.append(fraction_c1)
         # Construct the message     
         end_time = time.time()
         elapsed_time = end_time - start_time
         elapsed_time1 = round(elapsed_time, 0)
         formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
         if elapsed_time1 >= 60:
            min = elapsed_time1 / 60;       hours = min / 60
            progress_message = f"1: Progress: {round(100 * (t / t_brake), 1)}%. Use time: \
            {round(hours)} hours {round(min)} min. Start: {formatted_start_time }."
         else:
            progress_message = f"1: Progress: {round(100 * (t / t_brake), 1)}%. Use time: \
            {round(elapsed_time1)} s. Start: {formatted_start_time }."
         sys.stdout.write(f"\r{progress_message.ljust(80)}")  # 80 spaces to ensure full clearing
         sys.stdout.flush()

           
         #####################################
         co_ind, fa_mar, so_ind   = target_facets( domain, x_co_new, y_co_new, r_square  )
         facet_tag                = meshtags( domain, fdim, co_ind[ so_ind], fa_mar[ so_ind] )
         ds                       = Measure( "ds", domain=domain, subdomain_data=facet_tag)   

         F = ((rho * c) / dt[i] * inner(u, v) * dx
             + k * inner(grad(u), grad(v)) * dx
             + h * inner(u, v) * ds(b_con)
             + radiation * inner(u**4, v) * ds(b_con)
             - ( inner(f, v) * dx
                 + (rho * c) / dt[i] * inner(u_n, v) * dx  #!!!!!!!!!!!!   u_n need to double check
                 + h * Tm * v * ds(b_con)
                 + radiation * (Tm**4) * v * ds(b_con)) )
        
         g[i] = g[i]/ fraction_c1

         for j in list(range(1, 19)):
             #F += -k * dot(grad(u) * v, n_vector) * ds(10 * j) - inner(g[i], v) * ds(10 * j)
             F += ( - inner(g[i], v) * ds(10 * j) 
                    - h * inner( u, v) * ds(10 * j)  
                    - radiation * inner( (u**4 - Tm**4), v) * ds(10 * j) )    

         u_pre_solve = u.copy()         

         problem = NonlinearProblem(F, u, bcs=[bc])
         ## 7: Using petsc4py to create a linear solver
         solver_setup_solve(problem, u)
        
         u.x.scatter_forward()            
       
         # Update solution at previous time step (u_n)
         u_n.x.array[:] = u.x.array
         T_array.append((t, u.x.array.copy()))
         # Write solution to file
         xdmf.write_function(u, t)
         # Update plot
         #warped = grid.warp_by_scalar("uh", factor=0)
         plotter.update_coordinates(warped.points.copy(), render=False)
         plotter.update_scalars(u.x.array, render=False)
         plotter.write_frame()      

         print('Rub radius square is ', r_rub_new)

    plotter.close()
    xdmf.close()
    print()
    return(T_array, fraction_c, deformed_co, u_d1, d_wear  )

################################################################

def get_new_contact_nodes(x_co_zone, domain_pad, u_d, Vu, z1, x_co, y_co, S_rub_circle, i):
    import numpy as np
    # x_co_zone is the tolerance for contact range in z direction, z+ or z- x_contact_zone is range of contact condition
    gdim,fdim = 3,2
    original_co = domain_pad.geometry.x         # Initial coordinates of nodes (N x 3 array)
    u_d_vals    = u_d.x.array.reshape(-1, gdim) # Displacements (N x 3 array)
    deformed_co = original_co + u_d_vals        # Coordinates of deformed mesh

    low_contact_bool  =  (deformed_co[:, 2] < (z1+x_co_zone)) 
    high_contact_bool =  (deformed_co[:, 2] > (z1-x_co_zone)) 
    contact_boolean   =  low_contact_bool  &  high_contact_bool 
    contact_indices   =  np.where(contact_boolean)[0]  #contact indicies, from all nodes
    contact_co        =  original_co [ contact_indices ]
    
    if i == 0:
            S_rub_circle  = 353.44
            S_rub_circle1 = [S_rub_circle for _ in range(18) ] 
    else:
            S_rub_circle1 =  S_rub_circle   #S_rub_circle = r_rub**2 * c_contact
            #S_rub_circle1 = [353.44 for _ in range(18) ] 
    

    boundaries = []

    for j in range( 18): # boundaries include (marker, locator) 
            boundaries.append  ( lambda x,j=j: (x[0]-x_co[j])**2 +(x[1]-y_co[j])**2 <= S_rub_circle1[j])  
    contact_dofs = []  
    for j in range( 18):
            contact_dofs.append( fem.locate_dofs_geometrical(Vu, boundaries[j])  )
    ############################
    
    new_c_nodes = []
    for i in range( 18):
        contact_nodes_colum = deformed_co [contact_dofs [i]]  # column nodes, not only in surface
        tem_indi            = []
        new_contact         = []
        for j in range( len( contact_nodes_colum)):
            if ( contact_nodes_colum[j][2] <= (z1+x_co_zone) ) and ( contact_nodes_colum[j][2] >= (z1-x_co_zone) ):
                tem_indi = tem_indi + [j]   ## indices for contact_dofs, a column, only get surface index
            else:
                tem_indi = tem_indi 
          
        c1 = contact_dofs[i] [tem_indi] # get index for contact surface
        new_c_nodes = new_c_nodes + [c1]
    return deformed_co, new_c_nodes
    ############################

    