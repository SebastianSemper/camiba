import numpy as np
import camiba as cb

arr_d = np.array([2, 2])
arr_d = np.array([4])

u = np.arange(4).reshape(tuple(arr_d))+1
u = u + 1j*(4 - u)

utilde = np.copy(u)
utilde[0] = np.real(utilde[0])

T = cb.linalg.multilevel.Toep(u)
print("u\n", u)
print("T\n", T)
DT = cb.linalg.multilevel.ToepAdj(arr_d, T)
print("D(T)\n", DT)

W = cb.algs.admm._calc_w(arr_d)
print("W\n", W)

print("Wu \n", W*utilde.conj())
