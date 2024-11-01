# This file is the library functions for Disc calculatioin
# Author:      yanjun
# Start:       2024-03-01
# Last update: 2024-05-01
# Location:    Stockholm
# Institute:   KTH Royal Institute of TEchnology
# Github:      https://github.com/Yanjun96/FEniCSx.git

import dolfinx
print(f"DOLFINx version: {dolfinx.__version__}")

# import basic
import pyvista
import ufl
import dolfinx
import time
import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import meshio
import logging

# import speciail library
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
from dolfinx import fem, mesh, io, plot, default_scalar_type, nls, log
from dolfinx.fem import (
    Constant,
    dirichletbc,
    Function,
    FunctionSpace,
    form,
    locate_dofs_topological,
)
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner,
    Measure,
    dot,
    FacetNormal,
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI

from disc_f import *

start_time = time.time()
t = 0 #xdmf.write_function(uh, t)
from dolfinx import log

log.set_log_level(log.LogLevel.ERROR)  # Disable INFO and lower logs

print("Simulation environment setup complete.")

######################################################################################
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
def mesh_brake_disc(min_mesh, max_mesh, filename, mesh_type,pad_v_tag):   
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
    disc_v_tag = 31  #volume tag
    pad_v_tag  = 32  #volume tag
    volumes = gmsh.model.occ.getEntities(dim = 3)
    gmsh.model.addPhysicalGroup(3, volumes[0],  disc_v_tag)
    gmsh.model.addPhysicalGroup(3, volumes[1],  pad_v_tag)
    
    # Surfaces: brake disc, 21 = friction surface
    surfaces = gmsh.model.occ.getEntities(dim = 2)
    gmsh.model.addPhysicalGroup(2, (2,6), 21)
    
    # Rubbing elements, from 1 to 19, here 32 is the origin name tag of rub surface(32-49)
    rublist = list(range(pad_v_tag,pad_v_tag+18))
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

#########################33333333333333333#####################################################
def target_facets(domain,x_co,y_co,S_rub_circle):
    from dolfinx.mesh import locate_entities
    from dolfinx import mesh
    import numpy as np
    
    boundaries = []
    for j in range(18):
        boundaries.append  (  ( (j+1)*10, lambda x,j=j: (x[0]-x_co[j])**2 +(x[1]-y_co[j])**2 <= S_rub_circle[j])  )
 
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
    if len(D) == 0:
       facet_markers1 = []
    else:
        facet_markers1 = np.concatenate(D)

    ####################################   7
    b_con = 200
    boundary20 = (b_con, lambda x:  x[2] == 20)
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
    
#########################33333333333333333#####################################################
def target_facets_ini(domain,x_co,y_co,S_rub_circle_ini):
    from dolfinx.mesh import locate_entities
    from dolfinx import mesh
    import numpy as np
    
    boundaries = []
    for j in range(18):
        boundaries.append  (  ( (j+1)*10, lambda x,j=j: (x[0]-x_co[j])**2 +(x[1]-y_co[j])**2 <= S_rub_circle_ini)  )
 
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
    b_con = 200
    boundary20 = (b_con, lambda x:  x[2] == 20)
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
      ( dt, P, g, num_steps,  h,  radiation,  v_angular, Ti, Tm,   S_rub_circle,
       t, rho,  c,  k,  t_brake,   S_total ) = vehicle_initial(angular_r, v_vehicle, c_contact, c_acc)  
      print("1: Total tims is ", round(sum(dt), 2), "s")
      print("2: Total numb steps is ", num_steps)
      return (num_steps)

###################################################
def mesh_brake_all(mesh_min, mesh_max,pad_v_tag):
   import os
   from dolfinx.io import XDMFFile, gmshio
   from mpi4py import MPI  
   mesh_name = f"{mesh_min}-{mesh_max}"
   mesh_name1 = "m-{}.msh".format(mesh_name)
   mesh_name2 = "m-{}".format(mesh_name)
   logging.getLogger("gmshio").setLevel(logging.ERROR)

   if os.path.exists(mesh_name1):
     # Run this command if the file exists
     print(f"The file '{mesh_name1}' exists, start creat now:")
     domain, cell_markers, facet_markers = gmshio.read_from_msh(
         mesh_name1, MPI.COMM_WORLD, 0, gdim=3 )
   else:
    # Run this command if the file does not exist
     print(f"The file '{mesh_name1}' does not exist, start building:")
     mesh_brake_disc(mesh_min, mesh_max, mesh_name2, 'tetra',pad_v_tag)
     domain, cell_markers, facet_markers = gmshio.read_from_msh(
         mesh_name1, MPI.COMM_WORLD, 0, gdim=3 )

   return domain, cell_markers, facet_markers, mesh_name, mesh_name1, mesh_name2

###################################################
def project(function, space):
    from ufl import TestFunction, TrialFunction, dx, inner
    from dolfinx.fem.petsc import  LinearProblem
    
    u = TrialFunction(space)
    v = TestFunction(space)
    a = inner(u, v) * dx
    L = inner(function, v) * dx
    problem = LinearProblem(a, L, bcs=[])
    return problem.solve()

###################################################
def solver_setup_solve(problem,u):
  from mpi4py import MPI
  from dolfinx.fem.petsc import NonlinearProblem
  from dolfinx.nls.petsc import NewtonSolver
  from petsc4py import PETSc
  from dolfinx import log
    
  ## 7: Using petsc4py to create a linear solver
  solver = NewtonSolver(MPI.COMM_WORLD, problem)
  solver.convergence_criterion = "incremental"
  solver.rtol = 1e-6
  solver.report = True  # this make solver report show in a small window

  ksp = solver.krylov_solver
  opts = PETSc.Options()
  option_prefix = ksp.getOptionsPrefix()
  opts[f"{option_prefix}ksp_type"] = "cg"
  opts[f"{option_prefix}pc_type"] = "gamg"
  opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
  ksp.setFromOptions()

  log.set_log_level(log.LogLevel.ERROR)
  
  return solver.solve(u)

#######################################################
def plot_gif(V,u,gif_name):
   import matplotlib as mpl
   import pyvista
   from dolfinx import plot
    
   pyvista.start_xvfb()
   grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
   plotter = pyvista.Plotter()
   plotter.open_gif(gif_name, fps=30)
   grid.point_data["Temperature"] = u.x.array
   warped = grid.warp_by_scalar("Temperature", factor=0)
   viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    
   sargs = dict(
    title_font_size=25,
    label_font_size=20,
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1, )
    
   renderer = plotter.add_mesh( warped,
    show_edges=True,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    # clim=[0, max(uh.x.array)])
    clim=[0, 200], )
   return(plotter, sargs, renderer, warped, viridis, grid )

#######################################################
def initial_u_n(domain,Ti):
   from dolfinx import fem, default_scalar_type
   from dolfinx.fem import Function
   from disc_f import project
   import numpy as np
    # give the initial Temperature value to u_n
   V = fem.functionspace(domain, ("CG", 1)) # Define variational problem, CG is Lagrabge
   Q = fem.functionspace(domain, ("DG", 0)) # projected form Q onto V, DG is discontinuous        Lagrange. 
   T_init = Function(Q)  # T_init is a function, or a class
   T_init.x.array[:] = np.full_like(1, Ti, dtype=default_scalar_type)
   u_n = project(T_init, V)  # u_n is for initial condition and uh is the solver result.
   u_n.name = "u_n"

   return (V, T_init, u_n)

#######################################################
def mesh_setup(domain, V,mesh_name1,num_steps, angular_r, mesh_name2, c_contact,z_all, Tm, S_rub_circle_ini):
    
    fdim = domain.topology.dim - 1
    ## bc_disc is zero, no any dirichlete boundary condition, z = 100, not exist
    bc_disc = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], z_all*2)) 
    bc = fem.dirichletbc( PETSc.ScalarType(Tm), \
                         fem.locate_dofs_topological(V, fdim, bc_disc), V)
    logging.getLogger("meshio").setLevel(logging.ERROR)
    mesh_brake = meshio.read(mesh_name1)

    all_e = sum(len(cells.data) for cells in mesh_brake.cells)  # all_e is the total elements
    xdmf_name = "T-s-{}-d-{}-{}-c-{}-e-{}.xdmf".format( num_steps, angular_r, mesh_name2, c_contact, all_e)
    h5_name   = "T-s-{}-d-{}-{}-c-{}-e-{}.h5".format  ( num_steps, angular_r, mesh_name2, c_contact, all_e )
    xdmf = io.XDMFFile(domain.comm, xdmf_name, "w")
    xdmf.write_mesh(domain)

    x_co, y_co = get_rub_coordinate() # Create boundary condition
    co_ind, fa_mark, so_ind = target_facets_ini (domain, x_co, y_co, S_rub_circle_ini )
    
    facet_tag = meshtags (domain, fdim, co_ind[so_ind], fa_mark[so_ind] )
    ds = Measure("ds", domain=domain, subdomain_data=facet_tag)
    b_con = 200;

    return(fdim, bc, mesh_brake, all_e, xdmf, x_co,y_co, ds, b_con)
    
#######################################################

def variation_initial(V, T_init,domain, rho, c, b_con, radiation, h, k, xdmf,dt,ds,u_n, Tm,g,bc):
  
    uh = fem.Function(V)
    uh.name = "uh"
    uh = project(T_init, V)  ##give temperature to all elements
    xdmf.write_function(uh, t)

    # u = trial function, solution what we want to know
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(0))  ## heat source is 0
    n_vector = FacetNormal(domain)

    F = ( (rho * c) / dt[0] * inner(u, v) * dx
        + k * inner(grad(u), grad(v)) * dx
        + h * inner(u, v) * ds(b_con)  #b_con is name of contact surfaace, 
        + radiation * inner(u**4, v) * ds(b_con)
        - ( inner(f, v) * dx
           + (rho * c) / dt[0] * inner(u_n, v) * dx
           + h * Tm * v * ds(b_con)
           + radiation * (Tm**4) * v * ds(b_con) ) )

    for i in list(range(1, 19)):  # before 2024/5/16
        F += (+ inner(g[0], v) * ds(10 * i) 
              - h * inner( u, v) * ds(10 * i)  
              - radiation * inner( (u**4 - Tm**4), v) * ds(10 * i) )

    problem = NonlinearProblem(F, u, bcs=[bc])
    return (problem, u,v,f,n_vector)
     
#######################################################

def solve_heat(Ti, u, num_steps, dt, x_co, y_co, angular_r, \
               t_brake, domain, S_rub_circle, fdim,\
               rho, c, v, radiation, k, h, f, Tm, u_n, g,\
               ds, xdmf, b_con, bc, plotter, warped ):
    
    T_array = [(0, [Ti for _ in range(len(u.x.array))])]
    total_degree = 0
    t = 0
    for i in range(num_steps):
        t += dt[i]

        x_co, y_co = rub_rotation(x_co, y_co, angular_r)  # update the location
        total_degree += angular_r  # Incrementing degree by 10 in each step
        # Construct the message
        

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time1 = round(elapsed_time, 0)
        formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        if elapsed_time1 >= 60:
           min = elapsed_time1 / 60
           hours = min / 60
           progress_message = f"1: Progress: {round(100 * (t / t_brake), 1)}%. Use time: {round(hours)} hours {round(min)} min. Start: {formatted_start_time }."
        else:
           progress_message = f"1: Progress: {round(100 * (t / t_brake), 1)}%. Use time: {round(elapsed_time1)} s. Start: {formatted_start_time }."
    
        sys.stdout.write(f"\r{progress_message.ljust(80)}")  # 80 spaces to ensure full clearing
        sys.stdout.flush()

        co_ind, fa_mar, so_ind = target_facets(domain, x_co, y_co, S_rub_circle )
        facet_tag = meshtags( domain, fdim, co_ind[so_ind], fa_mar[so_ind] )
        ds = Measure("ds", domain=domain, subdomain_data=facet_tag)

        F = ((rho * c) / dt[i] * inner(u, v) * dx
            + k * inner(grad(u), grad(v)) * dx
            + h * inner(u, v) * ds(b_con)
            + radiation * inner(u**4, v) * ds(b_con)
            - ( inner(f, v) * dx
                + (rho * c) / dt[i] * inner(u_n, v) * dx
                + h * Tm * v * ds(b_con)
                + radiation * (Tm**4) * v * ds(b_con)) )

        for j in list(range(1, 19)):
            #F += -k * dot(grad(u) * v, n_vector) * ds(10 * j) - inner(g[i], v) * ds(10 * j)
            F += ( - inner(g[i], v) * ds(10 * j) 
                   - h * inner( u, v) * ds(10 * j)  
                   - radiation * inner( (u**4 - Tm**4), v) * ds(10 * j) )    

        problem = NonlinearProblem(F, u, bcs=[bc])

        ## 7: Using petsc4py to create a linear solver
        solver_setup_solve(problem,u)
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

    plotter.close()
    xdmf.close()
    print()
    return(T_array)
 
#######################################################
def plot_T_pad(domain_pad, T_pad):
    gdim,fdim = 3,2
    pyvista.start_xvfb()
    topology, cell_types, geometry = plot.vtk_mesh(domain_pad, gdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["Temperature/ °C"] = T_pad
    grid.set_active_scalars("Temperature/ °C")
    plotter = pyvista.Plotter()

    sargs = dict(title_font_size=25, label_font_size=20,  color="black",
             position_x=0.1, position_y=0.85, width=0.8, height=0.1)
    plotter.add_mesh(grid, show_edges=True,scalar_bar_args=sargs,clim=[0, 200])
    plotter.camera.azimuth = -5
    plotter.camera.elevation = 180  
    plotter.window_size = (800, 400)
    plotter.zoom_camera(1.5)
    return plotter

#######################################################
def plot_S_pad(Vu,u_d,scale_factor ):
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Vu)        # get mesh data
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry) # plot grid
    u_3D = np.zeros((u_geometry.shape[0], 3))
    u_3D[:, :3] = u_d.x.array.reshape(-1, 3)
    u_grid.point_data["Displacement/ mm"] = u_3D

    u_grid.set_active_vectors("Displacement/ mm")
    warped = u_grid.warp_by_vector("Displacement/ mm", factor=scale_factor )
    plotter = pyvista.Plotter()
    plotter.window_size = (800, 400)
    sargs = dict(title_font_size=25, label_font_size=20,  color="black",
             position_x=0.1, position_y=0.85, width=0.8, height=0.1)

    plotter.add_mesh(warped, scalar_bar_args=sargs)
    plotter.camera.azimuth = -5
    plotter.camera.elevation = 180  
    plotter.zoom_camera(1.5) 
    return plotter

###########################################################
def compare_two_arrays (array1, array2):
    ## this function is used to compare two different length arrays and find the different items.
    import numpy as np

    array1 = np.round(array1,2)
    array2 = np.round(array2,2)
    len1 = len(array1)
    len2 = len(array2)
    min_len = min(len1, len2)
    differences = []
    for i, item1 in enumerate(array1):
        found = False
        for j, item2 in enumerate(array2):
           if np.array_equal(item1, item2):
                found = True
                break
        if not found:
            differences.append((f"Array1 index {i}", item1, None))  # Not found in array2
    # Output the differences
    for desc, val1,val2 in differences:
        print(f"{desc}:")
        print(f"  Array 1: {val1}")
    if  len(differences) == 0:
        print("Old and new pad nodes does not have differences")

    return print()


###########################################################
def extract_u_n(mesh_name1, u_n, physical_group_tag):
    from mpi4py import MPI
    import numpy as np
    from dolfinx.io import gmshio
    import gmsh
    logging.getLogger("gmshio").setLevel(logging.ERROR)
    domain, cell_mark, facet_mark = gmshio.read_from_msh(mesh_name1, MPI.COMM_WORLD, 0, gdim=3)
    # cell_mark dim is 3, contains 31 and 32, physical name of brake disc and pad
    # facet_mark dim is 2,
    # domain is nodes, is less than cells, nearly n nodes = 3n cells.
    physical_cells = np.where(cell_mark.values == physical_group_tag)[0]
    cell_to_vertex = domain.topology.connectivity(domain.topology.dim, 0)
    physical_nodes = set() 
    # Loop through the physical cells and get the associated vertex (node) indices
    for cell in physical_cells:                 # cells in volume 32
         vertices = cell_to_vertex.links(cell)  # vertices in volume 32
         physical_nodes.update(vertices)        # added to physical_nodes
    # Convert the set of unique nodes to a sorted list
    physical_nodes = np.array(sorted(physical_nodes))  ## only part of the whole nodes
    # Extract the temperature values (u_n) corresponding to these nodes
    u_n_pad = u_n.x.array[physical_nodes]
    co_pad = domain.geometry.x[physical_nodes]
    return u_n_pad, physical_nodes, co_pad

###########################################################

def T_pad_transfer1(mesh_name1, mesh_n_pad, u_n, mesh_brake, pad_v_tag):
  
    
    # get pad coordinates from the whole brake mesh.
    T_old_pad , pad_nodes, co_pad = extract_u_n(mesh_name1, u_n, pad_v_tag)    
    from scipy.spatial import cKDTree
    logging.getLogger("gmshio").setLevel(logging.ERROR)
    domain_pad, cell_mark_pad, facet_mark_pad = gmshio.read_from_msh( mesh_n_pad , MPI.COMM_WORLD, 0, gdim=3 )
    
    #####  here is a function about map data according to coordinate, it seems do not need for now.2024-10-22.
    #tree = cKDTree(co_pad) # node_coordinates are from calculation
    #distances, indices = tree.query( domain_pad.geometry.x )
    #T_new_pad = np.zeros( len( T_old_pad))
    #for i, index in enumerate(indices):
    #    T_new_pad[i] = T_old_pad[index]  
    return T_old_pad, co_pad 

###########################################################

def mesh_del_disc(mesh_name1, mesh_n_pad):
    import gmsh    
    volume_to_delete = 31
    surface_to_delete = 21
    mesh_n_pad = "new_pad.msh"
    gmsh.initialize()
    gmsh.open(mesh_name1)
    physcical_groups = gmsh.model.get_physical_groups()
    for dim, tag in physcical_groups:
        if dim == 3:
            name = gmsh.model.getPhysicalName(dim,tag)
            print(f"Volume:{name}, Tag:{tag}")
            if tag == volume_to_delete:
                gmsh.model.removePhysicalGroups([(dim, tag)])
                #print(f"deleted wolume with tag {volume_to_delete}")
    for dim, tag in physcical_groups:
        if dim == 2:
            name = gmsh.model.getPhysicalName(dim,tag)
            if tag == surface_to_delete:
                gmsh.model.removePhysicalGroups([(dim, tag)])
                #print(f"deleted surface with tag {surface_to_delete}")
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_n_pad )
    gmsh.finalize()
    return mesh_n_pad
###########################################################
def contact_surface(domain_pad, facet_mark_pad): ## this contact area is only from mesh, not change with temperature.
    s_total = 0
    s_contact = []
    for i in range(1,19):
       contact_surface_marker_value = i  # Set this to the actual marker value for the contact surface
       # Ensure contact_measure is associated with the correct domain and facet marker
       contact_measure = ufl.ds(domain=domain_pad, subdomain_data=facet_mark_pad, subdomain_id=contact_surface_marker_value)
       # Define the integrand as a constant value `1.0` over the contact surface to represent area
       contact_area_form = 1.0 * contact_measure
       # Assemble the contact area by integrating over the contact surface
       contact_area = fem.assemble_scalar(fem.form(contact_area_form))
    s_contact.append(contact_area)
    s_total = s_total + contact_area
    print("Estimated Contact Area is:", round(sum(s_contact)/100*18, 2),"cm*2")
    return(s_total, s_contact)
###########################################################
def get_new_contact_nodes(x_co_zone, domain_pad, u_d, Vu, z1, x_co, y_co):
    # x_co_zone is the tolerance for contact range in z direction, z+ or z- x_contact_zone is range of contact condition
    gdim,fdim = 3,2
    original_co = domain_pad.geometry.x         # Initial coordinates of nodes (N x 3 array)
    u_d_vals    = u_d.x.array.reshape(-1, gdim) # Displacements (N x 3 array)
    deformed_co = original_co + u_d_vals        # Coordinates of deformed mesh

    low_contact_bool  =  (deformed_co[:, fdim] < (z1+x_co_zone)) 
    high_contact_bool =  (deformed_co[:, fdim] > (z1-x_co_zone)) 
    contact_boolean   =  low_contact_bool  &  high_contact_bool 
    contact_indices   =  np.where(contact_boolean)[0]  #contact indicies, from all nodes
    contact_co        =  original_co [ contact_indices ]
    #print('1: Minimum penalty deformation is ', min(u_d_vals[:,2]))
    #print('2: Length of contact indices is '  , len(contact_indices))
    #print('3: Length of u_d:  '               , len(deformed_co) )
    #print('4: Length of domain_pad: '         , len( original_co ) )
    #print('5: Length of contact_co: '         , len( contact_co  ) )
    
    #S_rub_circle = r_rub**2 * c_contact
    S_rub_circle = 1110.364507
    S_rub_circle1=[S_rub_circle for _ in range(18) ] 
    boundaries = []
    n_surface = len(S_rub_circle1)
    ## S_rub_circles1 should not change, it means the contact areas of rubbing elements, used to locate the boundaries.
    for j in range( n_surface): # boundaries include (marker, locator) 
            boundaries.append  ( lambda x,j=j: (x[0]-x_co[j])**2 +(x[1]-y_co[j])**2 <= S_rub_circle1[j])  
    contact_dofs = []  
    for j in range( n_surface):
            contact_dofs.append( fem.locate_dofs_geometrical(Vu, boundaries[j])  )
    ############################
    
    new_c_nodes = []
    for i in range( 18):
        contact_nodes_colum = deformed_co [contact_dofs [i]]  # cplumn nodes, not only in surface
        tem_indi            = []
        new_contact         = []
        for j in range( len( contact_nodes_colum)):
            if ( contact_nodes_colum[j][2] <= (z1+x_co_zone) ) and ( contact_nodes_colum[j][2] >= (z1-x_co_zone) ):
                tem_indi = tem_indi + [j]   ## indices for contact_dofs, a column, only get surface index
            else:
                tem_indi = tem_indi 
          
        c1 = contact_dofs[i][tem_indi] # get index for contact surface
        new_c_nodes = new_c_nodes + [c1]
    return deformed_co, new_c_nodes
###########################################################
def fit_circle(points):
    # points is an array, like 
    #[ [x1,y1]
    #  [x2,y2]
    #  [x3,y3] ]
    # Set up the matrix A and vector Z
    if len(points) == 0:
       h,k,r = 0,0,0
    else:
      A = np.column_stack((points[:, 0], points[:, 1], np.ones(points.shape[0])))
      Z = points[:, 0]**2 + points[:, 1]**2
    # Solve for [2h, 2k, C] using least-squares
      b, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
      h = b[0] / 2
      k = b[1] / 2
      C = b[2]
      # Calculate the radius
      r = np.sqrt(h**2 + k**2 + C)   
    return h, k, r
###########################################################
def get_r_xco_yco(deformed_co, new_c_nodes ):    
    x_co_new,y_co_new,r_rub_new1,r_rub_new = [],[],[],[]
    for i in range(18):
        z_nodes = []
        for j in range( len( deformed_co[ new_c_nodes[i]])):
            x = deformed_co[ new_c_nodes[ i]] [j] [0]
            y = deformed_co[ new_c_nodes[ i]] [j] [1]
            z_nodes[j:] =  [ [x,y] ]
        z_nodes    = np.array(z_nodes) 
        h1, k1, r1 = fit_circle(z_nodes)
        x_co_new   = x_co_new   + [h1]
        y_co_new   = y_co_new   + [k1]
        r_rub_new1 = r_rub_new1 + [r1] 

    no_0_r_rub = list(filter(lambda x: x!=0, r_rub_new1))  ## if contact points are 0, we do not add contact radius
    
    for i in range(18):
        if r_rub_new1[i] > 0: 
           r_rub_new.append( r_rub_new1[i]  + (18.8 - np.average(no_0_r_rub) ) )
        else:
           r_rub_new.append(0)
            
    for i in range(18):
        if r_rub_new[i] >= 19.5:
           r_rub_new[i] = 19.5
        else:
           r_rub_new[i] = r_rub_new[i]      
    
    r_rub_new = np.array(r_rub_new)
    S_rub_circle_new = np.pi * r_rub_new**2      #specific contact surfaces
    S_total_new = np.sum(S_rub_circle_new) /100  #overall all contact surfaces
    return(x_co_new, y_co_new,r_rub_new, S_total_new, S_rub_circle_new)
###########################################################

def penalty_method_contact(z1, Vu, u_d, aM, LM, u_, bcu ):
    
    def bottom_contact_nodes(x):
        return np.isclose(x[2], z1)
        
    contact_dofs = fem.locate_dofs_geometrical(Vu, bottom_contact_nodes)
    penalty_param = 400
    gdim = 3
    # Create a function to store penalty forces in the same function space as displacement
    penalty_forces = fem.Function(Vu)
    
    def update_penalty_force(u_d, penalty_forces, z1, penalty_param):
        u_vals = u_d.x.array.reshape(-1, gdim)
        penalty_forces_vals = penalty_forces.x.array.reshape(-1, gdim)
    # Apply penalty force for nodes below z1
        for dof in contact_dofs:
            if u_vals[dof][2] < 0:  ## here should <0 because contact surface is minus once expend with no constrain.
                penalty_forces_vals[dof][2] = -penalty_param * ( u_vals[dof][2]) # if here is not minus, rubing element grew up
            else:
                penalty_forces_vals[dof][2] = 0.0  # No penalty force if above z1
        penalty_forces.x.array[:] = penalty_forces_vals.ravel()

    update_penalty_force(u_d, penalty_forces, z1, penalty_param)
    LM_penalized = LM + ufl.inner(penalty_forces, u_) * ufl.dx
    problem = fem.petsc.LinearProblem(aM, LM_penalized, u=u_d, bcs=bcu)
    problem.solve()  
    return u_d
###########################################################
def T_S_deformation_solve (mesh_name1, u_n, mesh_brake, pad_v_tag, z4, ):

    gdim=3
    mesh_n_pad = mesh_del_disc(mesh_name1, "new_pad.msh")
    T_new_p, pad_node_coordinates  = T_pad_transfer1( mesh_name1, mesh_n_pad, u_n, mesh_brake, pad_v_tag )
    domain_pad, cell_mark_pad, facet_mark_pad = gmshio.read_from_msh( mesh_n_pad , MPI.COMM_WORLD, 0, gdim=3 )

    # defin the pad domain
    VT      = fem.functionspace(domain_pad, ("CG", 1))         #define the finite element function space
    Delta_T = fem.Function(VT, name ="Temperature_variation")  # T_ is the test function, like v
    for i in range(len(T_new_p)):
        Delta_T.vector.array[i] = T_new_p[i]

    #######try to make domain only for brake pad.
    E    = fem.Constant(domain_pad, 50e3)             # Elastic module
    nu   = fem.Constant(domain_pad, 0.2)              # Poission ratio
    gdim = domain_pad.geometry.dim

    mu    = E / 2 / (1 + nu)                          # Shear modulus
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)          # Lame parameters
    alpha = fem.Constant(domain_pad, 1e-5)            # Thermal expansion coefficient
    f1    = fem.Constant(domain_pad, (0.0, 0.0, 0.0)) # O for external force

    def eps(v):                                       # epsilon, strain, the deforamtion, dy/y 
        return ufl.sym(ufl.grad(v))
    def sigma(v, Delta_T):                            # sigmathis is sigma
        return (lmbda * ufl.tr(eps(v)) - alpha * (3 * lmbda + 2 * mu) * Delta_T 
        ) * ufl.Identity(gdim)  + 2.0 * mu * eps(v)   # here braces is important, can not be in above line

    Vu = fem.functionspace(domain_pad, ("CG", 1, (gdim,))) 
    du = ufl.TrialFunction(Vu)
    u_ = ufl.TestFunction(Vu)

    Wint = ufl.inner(sigma(du, Delta_T), eps(u_)) * ufl.dx  # here du is unkown
    aM   = ufl.lhs(Wint)                                    # Wint is long and lhs can help to distinguish unkown and know.
    LM   = ufl.rhs(Wint) + ufl.inner(f1, u_) * ufl.dx       # knows parameters are in lhs

    def up_side(x):
        return np.isclose(x[2], z4)

    up_dofs_u = fem.locate_dofs_geometrical(Vu, up_side)   # lateral sides of domain
    bcu       = [fem.dirichletbc(np.zeros((gdim,)), up_dofs_u, Vu)]  # displacement Vu is fixed in lateral sides

    u_d     = fem.Function(Vu, name="Displacement")
    problem = fem.petsc.LinearProblem(aM, LM, u=u_d, bcs=bcu)
    problem.solve()
    return u_d, Vu, aM, LM, bcu, u_, domain_pad 
###########################################################