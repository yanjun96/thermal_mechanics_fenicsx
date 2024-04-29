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


def mesh_brake_disc(min_mesh, max_mesh, filename):   
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

    ####################################
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
    
    ####################################
    common_indices3 = [common_indices1,common_indices2]
    facet_markers3  = [facet_markers1,facet_markers2]
    common_indices3 = np.concatenate(common_indices3)
    facet_markers3  = np.concatenate(facet_markers3)
    sorted_indices3 = np.argsort(common_indices3)

    return common_indices3, facet_markers3, sorted_indices3