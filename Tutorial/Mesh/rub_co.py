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

# Regular expression pattern to extract x and y coordinates
   pattern = r"addCylinder\(([\d.]+),\s*([\d.]+),"

# Find all matches in the text
   matches = re.findall(pattern, text)

# Extract x and y coordinates from matches
   x_co = [float(match[0]) for match in matches]
   y_co = [float(match[1]) for match in matches]
   return x_co, y_co