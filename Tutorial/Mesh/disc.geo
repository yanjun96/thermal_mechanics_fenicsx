// Parameters
lc = 0.1; // Characteristic length for meshing

// Define points
Point(1) = {0, 0, 0, lc};
Point(2) = {-1, 0, 0, lc};
Point(3) = {0, -1, 0, lc};
Point(4) = {1, 0, 0, lc};
Point(5) = {0, 1, 0, lc};
Point(6) = {0, 0, -0.1, lc}; // Thickness of the brake disc

// Define lines
Circle(7) = {2, 1, 3}; // Inner circle
Circle(8) = {3, 1, 4}; // Outer circle
Line(9) = {4, 1};
Line(10) = {5, 1};

// Extrude to create a 3D brake disc
Extrude {0, 0, 0.2} {
  Duplicata { Surface{7}; }
  Duplicata { Surface{8}; }
  Duplicata { Surface{9}; }
  Duplicata { Surface{10}; }
}

// Physical entities
Physical Surface("BrakeDisc") = {7, 8, 9, 10};

// Define the mesh
Mesh 3;

// Save the mesh
Save "brake_disc.msh";
