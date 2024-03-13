//+
Show "*";
//+
Show "*";
//+
Coherence;
//+
SetFactory("OpenCASCADE");
BooleanUnion{ Volume{3}; Delete; }{ Volume{50}; Delete; }
//+
BooleanUnion{ Volume{50}; Delete; }{ Volume{1}; Delete; }
//+
BooleanUnion{ Volume{3}; Delete; }{ Volume{50}; Delete; }
//+
SetFactory("OpenCASCADE");
BooleanUnion{ Volume{2}; }{ Volume{50}; }
//+
BooleanUnion{ Surface{6}; Delete; }{ Surface{44}; Delete; }
//+
Extrude {{0, 1, 0}, {0, 0, 0}, Pi/4} {
  Volume{50}; Layers{5}; Recombine;
}
//+
Symmetry {1, 0, 0, 1} {
  Duplicata { Volume{9}; Volume{8}; Volume{7}; Volume{10}; Volume{6}; Volume{5}; Volume{50}; Volume{2}; }
}
//+
Symmetry {1, 0, 0, 1} {
  Duplicata { Volume{50}; Volume{4}; }
}
//+
Symmetry {1, 0, 0, 1} {
  Duplicata { Volume{2}; }
}
//+
Symmetry {1, 0, 0, 1} {
  Duplicata { Volume{2}; }
}
//+
Physical Volume("1", 88) += {1};
//+
Physical Volume("1", 88) += {1};
