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