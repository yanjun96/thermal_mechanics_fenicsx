helix.msh is from gmsh.
helix.mesh is from salome.
helix.xdml and helix.xml is from convert: meshio convert helix.msh helix.xdmf

if use meshio convert helix.msh helix.dmf
it will create two xml files, geometrical and physical.xml

xml is a legacy format, now is xdmf instead.

2024-3-4

helix1.msh has changed the tag of the mesh
helix2.msh has changed the tag, 1 is body, 2 is out, 3 is inlet, 4 is volume, number is 16, which is work in fenics