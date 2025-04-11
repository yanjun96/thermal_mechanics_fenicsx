Author: Yanjun Zhang, PhD Student in Mechanical Engineering
Date: September 19, 2024
Location: KTH Royal Institute of Technology, Stockholm, Sweden

Overview
This repository is designed to model and calculate the temperature and deformation of thermo-mechanical coupled applications using FEM(finite element method). For now, the railway brake discs is the example because its strong thermo-mechanical effects. 


Methodology
The solution is based on the FEniCSx framework, which enables the efficient computation of coupled thermal and mechanical problems using FEM. This tool provides a powerful platform for solving complex partial differential equations, making it ideal for simulating the thermal and structural response of railway brake discs under operational loads.


Acknowledgements
Special thanks to the developers of FEniCSx for creating this invaluable tool that enables cutting-edge research in computational mechanics.


Usage

Step1: you need to install fenicsx first: https://fenicsproject.org/download/

conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista


Step 2: for only heat transfer analysis: use functions in /therm/disc_f.py

Step 3: for heat transfer + deformation + contact analysis, use functions in /mech/disc_f_wear.py

Step 4: you have to define your own mesh and markers, more details please see https://jsdokken.com/dolfinx-tutorial/, after you familiar with meshing, you can change your mesh to fit in this repository.


ps: this repository is not fully documented, it is ongoing work.
