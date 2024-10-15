# This file is the library functions for Disc calculatioin
# Author:      yanjun
# Start:       2024-03-01
# Last update: 2024-05-01
# Location:    Stockholm
# Institute:   KTH Royal Institute of TEchnology
# Github:      https://github.com/Yanjun96/FEniCSx.git


def vehicle_initial(angular_r, v_vehicle, c_contact, c_acc):
    import numpy as np
    v_ini = v_vehicle/3.6   /   (920/2/1000) 
    # D_wheel = 920 mm, v = D_wheel /2 /1000 * v_ini *3.6   # km/h
    # Start time, Final time  
    t = 0
    t_brake = 49
    t_lag = 4
    # rubbing element radius, Contact area 
    r_rub = 18.8
    S_rub_circle = r_rub**2 * c_contact
    S_total = S_rub_circle * np.pi * 18  #mm2
    # initial and brake pad temperature
    Ti = 60
    Tm = 60
    # density (kg.m^-3), capacity (J/Kg.K), conductivity (W/m.K)
    t_u = 1e3 # m to mm
    rho = 7850 /(t_u**3)
    c = 462
    k = 48 / t_u
    # mu, P_brake,  r_disc , heat_distribution  
    mu = 0.376
    P_initial = 274000
    r_disc = 0.25
    heat_distribution = 0.88
    # calculate total num_steps
    if c_acc == 1:  # constant acc for the whole process
        acc = v_ini/t_brake
        v_lag_end = (v_ini - (acc *t_lag) )   
        angular_r_rad = angular_r/180*np.pi  
        dt_lag = angular_r_rad  /  ( ( v_ini + v_lag_end  ) /2 )
        n_lag = round (t_lag / dt_lag) + 1 
        dt_a_lag = angular_r_rad  /  ( v_lag_end /2 )
        n_a_lag =  round ( (t_brake - t_lag) / dt_a_lag ) + 1
        num_steps = n_lag + n_a_lag
        dt = []
        v_angular = [v_ini]
        for i in range(num_steps):
            dt.append ( angular_r_rad / v_angular[i] )
            v_angular.append (  v_ini- sum(dt) * acc )           
        P = []
        for i in range(num_steps):
            if i <= n_lag:
                #P.append( P_initial/ n_lag * (i**(1/3)) )  ## no linear
                P.append( P_initial/ n_lag * (i**(1)) ) 
            else:
                P.append( P_initial) 
                
    else:  
        acc = v_ini/(t_brake-t_lag)
        v_lag_end = (v_ini - (acc *t_lag)*c_acc ) 
        acc_a_lag = v_lag_end / (t_brake-t_lag)
       
        angular_r_rad = angular_r/180*np.pi  
        dt_lag = angular_r_rad  /  ( ( v_ini + v_lag_end  ) /2 )
        # number of time step needed during lag
        n_lag = round (t_lag / dt_lag) + 1   
        dt_a_lag = angular_r_rad  /  ( v_lag_end /2 )
        n_a_lag =  round ( (t_brake - t_lag) / dt_a_lag ) + 1
        # number of time step needed after lag
        num_steps = n_lag + n_a_lag
        P = []
        for i in range(num_steps):
            if i <= n_lag:
                P.append( P_initial/ n_lag * i )
            else:
                P.append( P_initial) 
        dt = []
        v_angular = [v_ini]
        for i in range(num_steps):
            if i <= n_lag:
               dt.append ( angular_r_rad / v_angular[i] )
               v_angular.append (  v_ini-sum(dt)*acc*c_acc )
            else:
               dt.append ( angular_r_rad / v_angular[i] )
               v_angular.append (  v_lag_end- (sum(dt) - t_lag) * acc_a_lag )
        
    # S_or is the original brake pad rubbing area, 200 cm2. 
    S_or = 200
    S_new = S_total/100 #mm2 to cm2
    # g is the heat source,unit is w/mm2 
    g = []
    for i in range(num_steps):
        g.append ( mu * P[i] * v_angular[i] * r_disc * heat_distribution *2 /(t_u**2)  * (S_or/S_new) )
        
    #  h is the heat convection coefficient, unit is W/mm2 K  
    h = 7.75e-5
    # radiation is the radiation heat coefficient, unit is W/mm2 K
    # stefan-Boltzmann constant theta = 5.67*10e-8 w/m2 k-4,   0.64 is the emmissivity
    radiation = 5.670*(10e-8)/(t_u**2)  * 0.64

    return dt,P,g,num_steps,h,radiation,v_angular,Ti,Tm,S_rub_circle,t,rho,c,k,t_brake,S_total


#################################################################################################################  2
def rub_rotation(x, y, rotation_degree):
    import numpy as np
  
    # Define the rotation angle in radians (rotation_degree per second)
    rotation_radian = rotation_degree / 360 * 2 * np.pi
    angle = rotation_radian  # 1 radian

    # Define the rotation matrix
    r_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    
    points = np.vstack((x, y))

    # Perform the rotation
    r_points = r_matrix @ points

    # Separate the rotated x and y coordinates
    x1 = r_points[0, :]
    y1 = r_points[1, :]
    return x1, y1

#################################################################################################################  3
def get_rub_coordinate():
    import re
    # Sample text containing cylinder data
    text = """
    rub1  = gmsh.model.occ.addCylinder(214,27,z1,           0, 0, z2,  r_rub)
    rub2  = gmsh.model.occ.addCylinder(258,22,z1,           0, 0, z2,  r_rub)
    rub3  = gmsh.model.occ.addCylinder(252,63,z1,           0, 0, z2,  r_rub)
    rub4  = gmsh.model.occ.addCylinder(197, 66, z1,         0, 0, z2,  r_rub)
    rub5  = gmsh.model.occ.addCylinder(262, 105, z1,        0, 0, z2,  r_rub)
    rub6  = gmsh.model.occ.addCylinder(222,99, z1,          0, 0, z2,  r_rub)
    rub7  = gmsh.model.occ.addCylinder(240,148, z1,         0, 0, z2,  r_rub)
    rub8  = gmsh.model.occ.addCylinder(202,135, z1,         0, 0, z2,  r_rub)
    rub9  = gmsh.model.occ.addCylinder(168,111, z1,         0, 0, z2,  r_rub)
    rub10 = gmsh.model.occ.addCylinder(66.25,250.47,z1,     0, 0, z2,  r_rub)
    rub11 = gmsh.model.occ.addCylinder(138.27,146.38,z1,    0, 0, z2,  r_rub)
    rub12 = gmsh.model.occ.addCylinder(167.81,175.7, z1,    0, 0, z2,  r_rub)
    rub13 = gmsh.model.occ.addCylinder(187.21, 210.86, z1,  0, 0, z2,  r_rub)
    rub14 = gmsh.model.occ.addCylinder(135.83,201.65, z1,   0, 0, z2,  r_rub)
    rub15 = gmsh.model.occ.addCylinder(98.99,182.76, z1,    0, 0, z2,  r_rub)
    rub16 = gmsh.model.occ.addCylinder(105.58,237.44, z1,   0, 0, z2,  r_rub)
    rub17 = gmsh.model.occ.addCylinder(148.68,240, z1,      0, 0, z2,  r_rub)
    rub18 = gmsh.model.occ.addCylinder(63.53, 206.27, z1,   0, 0, z2,  r_rub)
    """
    x_coor = [214.0, 258.0, 252.0, 197.0, 262.0, 
             222.0, 240.0, 202.0, 168.0, 
             66.25, 138.27, 167.81, 187.21, 135.83, 
             98.99, 105.58, 148.68, 63.53]
    y_coor = [27.0, 22.0, 63.0, 66.0, 105.0,
             99.0, 148.0, 135.0, 111.0,
             250.47, 146.38, 175.7, 210.86, 201.65,
             182.76, 237.44, 240.0, 206.27]
    # Regular expression pattern to extract x and y coordinates
    pattern = r"addCylinder\(([\d.]+),\s*([\d.]+),"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Extract x and y coordinates from matches
    x_co = [float(match[0]) for match in matches]
    y_co = [float(match[1]) for match in matches]
    return x_co, y_co

#####################################################################################################################  4
def find_common_e(bcs, bcs_lists):
    # Create set for bcs
    
    set_bcs = set(tuple(bcs))
        # Initialize the union set with the set of bcs
    union = set()
    
    # Iterate through the list of lists
    for bc in bcs_lists:
        # Convert current list to set
        set_bc = set(tuple(bc))
        
        # Update the union set with the current list
        union = union.union(set_bc)
    
    # Find the common elements with bcs
    common_e = set_bcs.intersection(union)
    common_e_list = list(common_e)
    
    return common_e_list

#########################################################################################################################   5
def mesh_brake_disc(min_mesh, max_mesh, filename, mesh_type):   
    import gmsh
    import sys
    import math
    import os
    import numpy as np
    
    gmsh.initialize()
    
    # all the unit is mm
    # z1, z2, z3 is the height of brake disc, rubbing elemetn, pad lining in Z direction
    # rd_outer, rd_inner is for brake disc, rp_outer, rp_inner is for brake pad radiu. r_rub is for rubbing elements
    # angle1 is brake pad in degree system, angle is in radians system
    
    z1, z2, z3 = 20, 33, 30
    rd_o, rd_i = 320, 175 
    r_rub = 18.8
    rp_o, rp_i = 303, 178
    
    angle1 = 80
    angle = angle1 / 360 * 2 * math.pi
    
    # gmsh.model.occ.addCylinder
    # x (double), y (double), z (double), dx (double), dy (double), dz (double),
    # r (double), tag = -1 (integer), angle = 2*pi (double)
    
    # brake disc
    outer_disc  = gmsh.model.occ.addCylinder(0,0,0,  0, 0, z1,  rd_o)
    inner_disc  = gmsh.model.occ.addCylinder(0,0,0,  0, 0, z1,  rd_i)
    disk = gmsh.model.occ.cut([(3, outer_disc)], [(3, inner_disc)])
    
    # rubbing elements
    x_co, y_co = get_rub_coordinate()
    
    rub_list = []
    for i, (x, y) in enumerate(zip(x_co, y_co), start=1):
       var_name = f"rub{i}"
       tag = gmsh.model.occ.addCylinder(x, y, z1, 0, 0, z2, r_rub)
       globals()[var_name] = tag
       rub_list.append(globals()[var_name])
    
    # brake pad, in [(3, )],3 means dimension, cut the common place, out - inner
    outer_pad  = gmsh.model.occ.addCylinder(0,0,z1+z2,  0, 0, z3,  rp_o, 50, angle)
    inner_pad  = gmsh.model.occ.addCylinder(0,0,z1+z2,  0, 0, z3,  rp_i, 51, angle)
    pad = gmsh.model.occ.cut([(3, outer_pad)], [(3, inner_pad)]) 
    
    # Initialize the shell with the first rub
    rub_list = [rub1, rub2, rub3, rub4, rub5, rub6, rub7, 
                rub8, rub9, rub10, rub11, rub12, rub13, rub14, rub15, rub16, rub17, rub18]
    shell = gmsh.model.occ.fuse([(3, outer_pad)], [(3, rub_list[0])], 70)
    for i in range(len(rub_list) - 1):
        shell = gmsh.model.occ.fuse([(3, 70 + i)], [(3, rub_list[i + 1])], 71 + i)
    gmsh.model.occ.synchronize()
    
    # Add physical group, this step should after synchronize to make sure success
    # https://gitlab.onelab.info/gmsh/gmsh/blob/master/tutorials/python/t1.py#L115
    
    # Volumes: 31,32 brake disc and pad.
    volumes = gmsh.model.occ.getEntities(dim = 3)
    gmsh.model.addPhysicalGroup(3, volumes[0],  31)
    gmsh.model.addPhysicalGroup(3, volumes[1],  32)
    
    # Surfaces: brake disc, 21 = friction surface
    surfaces = gmsh.model.occ.getEntities(dim = 2)
    gmsh.model.addPhysicalGroup(2, (2,6), 21)
    
    # Rubbing elements, from 1 to 19, here 32 is the origin name tag of rub surface(32-49)
    rublist = list(range(32,50))
    for rub in rublist:
        gmsh.model.addPhysicalGroup(2, (2, rub), rub-31)

    if mesh_type == 'hexa':
       gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
    
    # for the rubbing elements, P13 of UIC 541-3
    # Sinter material, 200 cm2, 18 rubbing elemets, r = 1.88 cm
    # Mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh)
    
    # Mesh file save
    gmsh.model.mesh.generate(3)
    c = gmsh.write(filename + ".msh")
    notice = print("NOTICE:" + filename + " has been meshed successfully and saved as " + filename + ".msh")   
  
    gmsh.finalize()
    
    return c
    return notice

#########################33333333333333333###########################################################################  6
def mesh_brake_pad(min_mesh, max_mesh, filename, mesh_type):   ##unfinished
    import gmsh
    import sys
    import math
    import os
    import numpy as np
    
    gmsh.initialize()
    
    # all the unit is mm
    # z1, z2, z3 is the height of brake disc, rubbing elemetn, pad lining in Z direction
    # rd_outer, rd_inner is for brake disc, rp_outer, rp_inner is for brake pad radiu. r_rub is for rubbing elements
    # angle1 is brake pad in degree system, angle is in radians system
    
    z1, z2, z3 = 20, 33, 30
    rd_o, rd_i = 320, 175 
    r_rub = 18.8
    rp_o, rp_i = 303, 178
    
    angle1 = 80
    angle = angle1 / 360 * 2 * math.pi
    
    # gmsh.model.occ.addCylinder
    # x (double), y (double), z (double), dx (double), dy (double), dz (double),
    # r (double), tag = -1 (integer), angle = 2*pi (double)
    
    # brake disc
    outer_disc  = gmsh.model.occ.addCylinder(0,0,0,  0, 0, z1,  rd_o)
    inner_disc  = gmsh.model.occ.addCylinder(0,0,0,  0, 0, z1,  rd_i)
    disk = gmsh.model.occ.cut([(3, outer_disc)], [(3, inner_disc)])
    
    # rubbing elements
    x_co, y_co = get_rub_coordinate()
    
    rub_list = []
    for i, (x, y) in enumerate(zip(x_co, y_co), start=1):
       var_name = f"rub{i}"
       tag = gmsh.model.occ.addCylinder(x, y, z1, 0, 0, z2, r_rub)
       globals()[var_name] = tag
       rub_list.append(globals()[var_name])
    
    # brake pad, in [(3, )],3 means dimension, cut the common place, out - inner
    outer_pad  = gmsh.model.occ.addCylinder(0,0,z1+z2,  0, 0, z3,  rp_o, 50, angle)
    inner_pad  = gmsh.model.occ.addCylinder(0,0,z1+z2,  0, 0, z3,  rp_i, 51, angle)
    pad = gmsh.model.occ.cut([(3, outer_pad)], [(3, inner_pad)]) 
    
    # Initialize the shell with the first rub
    rub_list = [rub1, rub2, rub3, rub4, rub5, rub6, rub7, 
                rub8, rub9, rub10, rub11, rub12, rub13, rub14, rub15, rub16, rub17, rub18]
    shell = gmsh.model.occ.fuse([(3, outer_pad)], [(3, rub_list[0])], 70)
    for i in range(len(rub_list) - 1):
        shell = gmsh.model.occ.fuse([(3, 70 + i)], [(3, rub_list[i + 1])], 71 + i)
    gmsh.model.occ.synchronize()
    
    # Add physical group, this step should after synchronize to make sure success
    # https://gitlab.onelab.info/gmsh/gmsh/blob/master/tutorials/python/t1.py#L115
    
    # Volumes: 31,32 brake disc and pad.
    volumes = gmsh.model.occ.getEntities(dim = 3)
    gmsh.model.addPhysicalGroup(3, volumes[0],  31)
    gmsh.model.addPhysicalGroup(3, volumes[1],  32)
    
    # Surfaces: brake disc, 21 = friction surface
    surfaces = gmsh.model.occ.getEntities(dim = 2)
    gmsh.model.addPhysicalGroup(2, (2,6), 21)
    
    # Rubbing elements, from 1 to 19, here 32 is the origin name tag of rub surface(32-49)
    rublist = list(range(32,50))
    for rub in rublist:
        gmsh.model.addPhysicalGroup(2, (2, rub), rub-31)

    if mesh_type == 'hexa':
       gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
    
    # for the rubbing elements, P13 of UIC 541-3
    # Sinter material, 200 cm2, 18 rubbing elemets, r = 1.88 cm
    # Mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh)
    
    # Mesh file save
    gmsh.model.mesh.generate(3)
    c = gmsh.write(filename + ".msh")
    notice = print("NOTICE:" + filename + " has been meshed successfully and saved as " + filename + ".msh")   
  
    gmsh.finalize()
    
    return c
    return notice

#########################33333333333333333###########################################################################
def target_facets(domain,x_co,y_co,S_rub_circle):
    from dolfinx.mesh import locate_entities
    from dolfinx import mesh
    import numpy as np
    
    boundaries = []
    for j in range(18):
        boundaries.append  (  ( (j+1)*10, lambda x,j=j: (x[0]-x_co[j])**2 +(x[1]-y_co[j])**2 <= S_rub_circle)  )
 
    facet_indices1, facet_markers1 = [], [] 
    fdim = 2 
    for (marker, locator) in boundaries:
        facets = locate_entities(domain, fdim, locator)   
        facet_indices1.append(facets)
        facet_markers1.append(np.full_like(facets, marker))
    facet_indices1 = np.hstack(facet_indices1).astype(np.int32)
    facet_markers1 = np.hstack(facet_markers1).astype(np.int32)
            
    A1 = facet_indices1
    B  = facet_markers1 
    C  = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], 20) )

    common_indices1 = np.intersect1d(A1,C)
    D = []
    for index in common_indices1:
        rows_A1 = np.where(A1 == index)
        D.append( B[rows_A1] )
    facet_markers1 = np.concatenate(D)

    ####################################   7
    boundary20 = (200, lambda x:  x[2] == 20)
    facet_indices2, facet_markers2 = [], [] 
    fdim = 2  
    for (marker, locator) in [boundary20]:
        facets = locate_entities(domain, fdim, locator)   
        facet_indices2.append(facets)
        facet_markers2.append(np.full_like(facets, marker)) 
    facet_indices2 = np.hstack(facet_indices2).astype(np.int32)
    facet_markers2 = np.hstack(facet_markers2).astype(np.int32)

    A1 = facet_indices2
    B  = facet_markers2
    B1 = common_indices1
    common_indices2 = np.setdiff1d(A1,B1)
    D  = []
    for index in common_indices2:
        rows_A1 = np.where(A1 == index)
        D.append( B[rows_A1] )
    facet_markers2 = np.concatenate(D) 
    
    ####################################   8
    common_indices3 = [common_indices1,common_indices2]
    facet_markers3  = [facet_markers1,facet_markers2]
    common_indices3 = np.concatenate(common_indices3)
    facet_markers3  = np.concatenate(facet_markers3)
    sorted_indices3 = np.argsort(common_indices3)

    return common_indices3, facet_markers3, sorted_indices3
#######################################################################################################################   9
def read_msh_nodes(filename):
    nodes = []
    nodes_c = []
    node_tag = []
    reading_nodes = False
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('$Nodes'):
                reading_nodes = True             
                continue
            elif line.startswith('$EndNodes'):
                reading_nodes = False             
                break
            elif reading_nodes:
                parts = line.split()         
                if len(parts) == 1:  # This line contains only node tag
                    node_tag.append ( int(parts[0]) )
                elif len(parts) == 3:  # This line contains node coordinates
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    nodes_c.append((x, y, z))
    for i in range(len(node_tag)):
        nodes.append( (node_tag[i], nodes_c[i])  )
    
    return nodes,node_tag
###############################################################################################################  10
def filter_nodes_by_z(nodes, z_value):
    filtered_nodes = [node for node in nodes if node[1][2] == z_value]
    return filtered_nodes

##############################################################################################################  11
def got_T_check_location(A1):
    ## A1 should like [247.5, 0]
    import numpy as np
    z = 19
    A2_b =  (  (A1[0]-40), 0 )
    A3_b =  (  (A1[1]+40), 0 )

    x = np.array(A1[0])
    y = np.array(A1[1])
    # Define the rotation angle in radians (1 radian per second)
    r_points = []
    for i in [0,1]:
      angle = 120*(i+1)   / 180 * np.pi
     
      # Define the rotation matrix
      r_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
      # Stack x and y into a single array of shape (2, n) where n is the number of points
      points = np.vstack((x, y))

      # Perform the rotation
      r_points.append( r_matrix @ points )

    # Separate the rotated x and y coordinates
    A2 = r_points[0]
    A3 = r_points[1]
    A2_fin = ( round( A2[0][0],2) , round(A2[1][0],2), z)
    A3_fin = ( round( A3[0][0],2) , round(A3[1][0],2), z)
    A1_fin = ( round( A1[0],1) , round(A1[1],2), z)
    #print( "A1 location is ",A1_fin, 
         #"\nA2 location is ",A2_fin, 
         #"\nA3 location is ",A3_fin)

    return A1_fin, A2_fin, A3_fin
##################################################################################################################  12

def save_t_T (csv_name, T_array):
    import csv
    t = []
    T = []
    for value in T_array:
        t.append(value[0])
        T.append(value[1])   
    # Specify the file path
    file_path = csv_name  # File path for CSV file
    # Write t and T to a CSV file
    with open(file_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["t", "T"])  # Write header
        for t_value, T_values in zip(t, T):
            for T_value in T_values:
                csv_writer.writerow([t_value, T_value])  # Write each value of t and corresponding value(s) of T

    # Confirmation message
    print("t and T have been successfully saved as", file_path)

#####################################################################################################################  13
def read_t_T (csv_name):
    ## csv_name = "xxxxx.csv"

    import csv
    from collections import defaultdict

    # Specify the file path
    file_path = csv_name  # File path for CSV file

    # Initialize a dictionary to store t and corresponding values of T
    t_T_dict = defaultdict(list)

    # Read t and T from the CSV file
    with open(file_path, "r", newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            t_value = float(row[0])  # Assuming t values are floats
            T_value = float(row[1])  # Assuming T values are floats
            t_T_dict[t_value].append(T_value)

    # Extract unique values of t and corresponding values of T
    t1 = list(t_T_dict.keys())
    T1 = [t_T_dict[t_value] for t_value in t1 ]

    # Confirmation message
    print("t and T have been successfully extracted from", file_path)
    return (t1,T1)

#####################################################################################################################  14

def find_3_coord(filename):
    import numpy as np
    
    ## below labels should always add if new mesh has result
    coord_lib = {'m-1-15.msh': [2201, 1590, 260 ],
                 'm-3-10.msh': [3157, 7018, 2141],
                 'm-3-15.msh': [2201, 1590, 260],
                 'm-3-7.msh':  [12266, 11501, 617],
                 'm-3-5.msh':  [19098, 34079, 7351],
                 'm-3-3.msh':  [94411, 114209, 8995],
                 'm-1-2.msh':  [333431, 308947, 18936],
                 'm-3-2.msh':  [333431, 308947, 18936],
                 'm-3-20.msh': [1713, 1587, 708] }
                                
    
    if filename in coord_lib:
        print('Lables already exists, for mesh',filename, "is ", coord_lib[filename])
        return coord_lib[filename]
    
    else:
        nodes = []
        nodes_c = []
        closest_coordinate = []
        node_tag = []
        reading_nodes = False
        with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('$Nodes'):
                        reading_nodes = True             
                        continue
                    elif line.startswith('$EndNodes'):
                        reading_nodes = False             
                        break
                    elif reading_nodes:
                        parts = line.split()         
                        if len(parts) == 1:  # This line contains only node tag
                            node_tag.append ( int(parts[0]) )
                        elif len(parts) == 3:  # This line contains node coordinates
                            x = float(parts[0])
                            y = float(parts[1])
                            z = float(parts[2])
                            nodes_c.append((x, y, z))
        for i in range(len(node_tag)):
                nodes.append( (node_tag[i], nodes_c[i])  )

        A1_fin, A2_fin, A3_fin = got_T_check_location([247.5, 0])
        Three_points = [A1_fin, A2_fin, A3_fin]
        for target in  Three_points:
            A_point = target
            coordinates = nodes_c
            distances = [np.sqrt((x - A_point[0])**2 + (y - A_point[1])**2 
                         + (z - A_point[2])**2) for x, y, z in coordinates]
            closest_index = np.argmin(distances)
            closest_coordinate.append(  coordinates[closest_index] )

        # Print the closest coordinate
        print("Closest coordinate is \n",
              tuple(round(coord, 2) for coord in closest_coordinate[0]),
             "\n", tuple(round(coord, 2) for coord in closest_coordinate[1]),
             "\n", tuple(round(coord, 2) for coord in closest_coordinate[2]),
             "\nPlease open the xdmf file in paraview, and find the labels for above three nodes and input as",
             "\nT_3_labels = [label1, label2, label3]. \nPlease also add in labels dictionary, functions in disc_f.py ")

#######################################
def find_3_coord_hexa(filename):
    import numpy as np
    
    ## below labels should always add if new mesh has result
    ## ONLY for hexahedral
    #coord_lib = {'m-3-5.msh':  [19098, 34079, 7351],
               #  'm-3-10.msh': [3157, 7018, 2141],
               #  'm-3-15.msh': [2201, 1590, 260], 
               #  'm-3-20.msh': [1713, 1587, 708] }
    
    coord_lib = {'m-3-10.msh': [53970, 122401, 36114],
                 'm-3-20.msh': [17818, 1136, 15718] }    
                                
    
    if filename in coord_lib:
        print('Lables already exists, for mesh',filename, "is ", coord_lib[filename])
        return coord_lib[filename]
    
######################################  15
def collect_csv_files(directory):
    import os
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

########################################  16

def extract_mesh_labels(file_name):
    import re
    match = re.search(r'-m-(\d+-\d+)', file_name)
    if match:
        mesh_labels = match.group(1)
        return f'm-{mesh_labels}.msh'
    else:
        return None

##########################################  17

def extract_file_labels(file_name, type_is):
    import re
    if type_is == 'mesh_size':
       match = re.search(r'e-(\d+)', file_name)
       if match:
          return int(match.group(1))
       return 0 
    if type_is == 'time_step':
       match = re.search(r's-(\d+)', file_name)
       if match:
          return int(match.group(1))
       return 0 
    if type_is == 'contact_area':
       match = re.search(r'c-(\d+(\.\d+)?)', file_name)
       if match:
          return float(match.group(1))
       return 0 

################################################
def add_indentation(old_notebook, new_notebook):
    ## example: add_indentation('Disc4_Concise.py', 'Disc4_Concise2.py' ) 
    with open(old_notebook, 'r') as f:
        lines = f.readlines()
    indented_lines = ['      ' + line for line in lines]
    with open(new_notebook, 'w') as f:
        f.writelines(indented_lines)


##############################################
def get_time_step_from_angular(angular2,mesh_max2,c_contact2):
      # import basic   
  
      import numpy as np  
      
      # import own functions
      from disc_f import vehicle_initial    
          
      # mesh-size, contact area coefficient
      mesh_min = 3
      mesh_max = mesh_max2
      c_contact = c_contact2
      
      # Each time step rotation angular, and acc during lag, 1 is full acc, 0 is no acc.
      angular_r = angular2
      v_vehicle = 160
      c_acc = 1
      
      # calling local functions to get all parameters
      ( dt, P, g, num_steps,  h,  radiation,  v_angular, Ti, Tm,    S_rub_circle,
       t, rho,  c,  k,  t_brake,   S_total ) = vehicle_initial(angular_r, v_vehicle, c_contact, c_acc)  
      print("1: Total tims is ", round(sum(dt), 2), "s")
      print("2: Total numb steps is ", num_steps)
      return (num_steps)