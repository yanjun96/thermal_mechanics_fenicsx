def mesh_brake_disc(min_mesh, max_mesh, filename ):
    
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
    import sys
    sys.path.append('/home/yanjun/Documents/FEniCSx/Tutorial/Mesh/')
    from rub_co import get_rub_coordinate
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
    # Launch the GUI to see the results:
    # if '-nopopup' not in sys.argv:
    #    gmsh.fltk.run()
    gmsh.finalize()
    
    return c
    return notice