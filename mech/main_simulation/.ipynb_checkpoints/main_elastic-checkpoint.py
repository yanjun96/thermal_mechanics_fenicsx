from disc_f import *
def main_elastic(pa, type, angular1, mesh_max1, c_contact1):
   

   # type should ""time_step"" , "mesh_size" , "contact_area"
   # angular1,  is degree per time step, 
   # mesh_max1, is between 1-15   mm
   # c_contact1 is between 0-1  
   ## C = [188, 259, 445]
           
   for parameter_c in pa:
      if type == "time_step" :
           
            angular2 = parameter_c
            mesh_max2 = mesh_max1
            c_contact2 = c_contact1
      if type == "mesh_size" :
        
            angular2 = angular1
            mesh_max2 = parameter_c
            c_contact2 = c_contact1
      if type == "contact_area" :
        
            angular2 = angular1
            mesh_max2 = mesh_max1
            c_contact2 = parameter_c       
         
      #!/usr/bin/env python
      # coding: utf-8
      
      # Author: yanjun zhang
      # ## Concise
      # ## 1: Start
      # Source from book "Abali - 2017 - Computational Reality" P119
#
      
      # In[1]:     
      
      # import basic
      ######################## 
      #from disc_f import *
      
      # calculate how long time the simulation it is
      start_time = time.time()
      
      # mesh-size, contact area coefficient
      mesh_min = 3
      mesh_max = mesh_max2
      c_contact = c_contact2

      # Each time step rotation angular, and acc during lag, 1 is full acc, 0 is no acc.
      angular_r    = angular2
      v_vehicle, c_acc   = 160, 1
      z1,z2,z3,z4,z_all  = 20,33,30,83,8
      pad_v_tag          = 32
      alpha_thermal      = 1.5e-6 #thermal expansion coefficient
      penalty_param      = 400
      k_wear             = 6.7e-6/10
      wear_f             = 'on'   # on is open for wear calcualtion on deformation 
       
      
      (dt, P, g, num_steps, h, radiation, v_angular, \
      Ti, Tm, S_rub_circle_ini, t, rho, c, k, t_brake, 
      S_total,)                                       = vehicle_initial (angular_r, v_vehicle,  c_contact, c_acc)
      print("1: Total braking tims is ", round(sum(dt), 2), "s")
      print("2: Total numb steps is ", num_steps)

      ## here use lots of abbreviation, details are in disc_f
      domain, cell_markers, facet_markers, mesh_name, mesh_name1, mesh_name2 \
                       = mesh_brake_all(mesh_min,mesh_max,pad_v_tag)

      V, T_init, u_n         = initial_u_n(domain, Ti)

      fdim, bc, mesh_brake, all_e,xdmf, x_co, y_co, ds, b_con \
                       = mesh_setup( domain, V, mesh_name1, num_steps, \
                         angular_r, mesh_name2, c_contact, z_all, Tm, S_rub_circle_ini)
      # Initialize
      problem, u, v, f, n_vector = variation_initial(V, T_init, domain, rho, c, b_con,\
                               radiation, h, k, xdmf, dt, ds, u_n, Tm,g,bc);
      solver_setup_solve(problem,u)

      ## Visualization of time dependent problem using pyvista
      gif_name    = "T-s-{}-d-{}-{}-c-{}-e-{}.gif".format(num_steps, angular_r, mesh_name2, c_contact, all_e)
      plotter, sargs, renderer, warped, viridis, grid = plot_gif(V,u,gif_name)
      ##solve
      num_steps= int(num_steps)

      x_co_zone   = 0.0001

      T_array,fraction_c,deformed_co,u_d1, d_wear   = solve_heat(Ti, u, num_steps, dt, x_co, y_co, angular_r, \
               t_brake, domain, S_rub_circle_ini, fdim,\
               rho, c, v, radiation, k, h, f, Tm, g,\
               ds, xdmf, b_con, bc, plotter, warped,\
               mesh_name1, mesh_brake, pad_v_tag, z4,\
               z1, x_co_zone, u_n, alpha_thermal, penalty_param, P, k_wear, wear_f)  #last u         should be u_n, here we set u, \
      #the same with previous:solver_setup_solve(problem,u)
      ####################################################################

      csv_name    = "Result_T-s-{}-d-{}-{}-c-{}-e-{}.csv".format(num_steps, angular_r,mesh_name2, c_contact, all_e)
      save_t_T(csv_name, T_array) # got the Temperature data



# # 6: Post process     
      

      
      end_time = time.time()
      elapsed_time = end_time - start_time
      elapsed_time1 = round(elapsed_time, 0)
      
      formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
      formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
      
      mesh_name = f"{mesh_min}-{mesh_max}"
      mesh_filename1 = "m-{}.msh".format(mesh_name)
      mesh_filename2 = "m-{}".format(mesh_name)
       
      txt_name = "T-s-{}-d-{}-{}-c-{}-e-{}.txt".format(num_steps, angular_r, mesh_filename2, c_contact, all_e )
       
      with open(txt_name, "w") as file:
          file.write("1: Start at: {}\n".format(formatted_start_time))
          file.write("2: End at: {}\n".format(formatted_end_time))  
          if elapsed_time1 >= 60:
              min = elapsed_time1 / 60
              hours = min / 60
              file.write("3: Simulation time is {} hours {} minutes\n".format(round(hours), round(min)))
          else:
              file.write("3: Simulation time is {} second\n".format(elapsed_time1))
          file.write("4: First time step dt is {} s\n".format(round(dt[0], 5)))
          r_disc = 0.25
          file.write("5: Convection heat transfer coefficient is {} W/mm2 K\n".format(h))
          file.write("6: Radiation is {} W/mm2 K-4\n".format(round(radiation, 14)))
          file.write("7: Each rotation degree is {} per time step or {} circle\n".format(round(angular_r, 1), round(angular_r / 360, 1)))
          file.write("8: The first rotation degree is {}\n".format(round(v_angular[0] * r_disc * dt[0] * 1000, 1)))
          file.write("9: The mid rotation degree is {}\n".format(round(v_angular[round(num_steps / 2)] * r_disc * dt[round(len(dt) / 2)] * 1000, 1)))
          file.write("10: The last rotation degree is {}\n".format(round(v_angular[num_steps - 1] * r_disc * dt[-1] * 1000, 1)))
          file.write("11: Total contact area of 18 rubbing element is {} cm2\n".format(round(S_total / 100, 1)))
          file.write("12: The mesh element size is between {}-{} mm\n".format(mesh_min, mesh_max))
          file.write("13: Total elements number is {}\n".format(all_e))
                         
      with open(txt_name, "r") as file1:
          print(file1.read())
      
      #### move files
      # Define the source directory
      source_dir = "/home/yanjun/documents/fenicsx/mech_disc/main_simulation"
   
      # Define the destination directory
      if type == "time_step" :
          destination_dir = "/home/yanjun/documents/sim_results/elasticity_2025/dt"

      if type == "mesh_size" :
          destination_dir = "/home/yanjun/documents/sim_results/elasticity_2025/dx"

      if type == "contact_area" :
          destination_dir = "/home/yanjun/documents/sim_results/elasticity_2025/dc"
    
      
      # Create the new folder in the destination directory
      new_folder_name = f"s-{num_steps}-d-{angular_r}-m-{mesh_min}-{mesh_max}-c-{c_contact}-e-{all_e}"
      destination_dir = os.path.join(destination_dir, new_folder_name)
      os.makedirs(destination_dir, exist_ok=True)

      h5_name = "T-s-{}-d-{}-{}-c-{}-e-{}.h5".format(num_steps, angular_r, mesh_filename2, c_contact, all_e) 
      xdmf_name = "T-s-{}-d-{}-{}-c-{}-e-{}.xdmf".format(num_steps, angular_r, mesh_filename2, c_contact, all_e )
          
      # List of files to move
      files_to_move = [gif_name, h5_name, xdmf_name, txt_name, csv_name]
      # Move each file to the destination directory
      for filename in files_to_move:
          source_file_path = os.path.join(source_dir, filename)
          destination_file_path = os.path.join(destination_dir, filename)
          shutil.move(source_file_path, destination_file_path)
      print("Move files successfully")   
