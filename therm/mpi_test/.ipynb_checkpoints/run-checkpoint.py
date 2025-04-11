import dolfinx

from main_thermal import main_thermal
# c is the parameter, which will change
para = [1350]
angular = 36980
mesh_max = 15
c_contact = 1
type = 'time_step'

main_thermal(para, type ,angular, mesh_max, c_contact)
