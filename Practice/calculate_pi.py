def calculte_pi(n_num):

  import random
  n_circul = 0
  for i in range(n_num):
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    if x**2 + y**2 <= 1:
      n_circul += 1
  pi_c = 4 * (n_circul/n_num)
  print("Total numer is {n_num}, n_circle is {n_circul}")

  return pi_c
