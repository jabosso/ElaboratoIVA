from scipy import linalg as la
import numpy as np





def gram_d(pose_1,pose_2):
    g1 = np.outer(pose_1, pose_1.T)
    g2 = np.outer(pose_2, pose_2.T)
    print(g1)


    d = np.trace(g1)+ np.trace(g2)-2.0*np.trace(la.fractional_matrix_power(np.dot(la.fractional_matrix_power(g1,1.5),
                                                                                  np.dot(g2,la.fractional_matrix_power(g1,0.5))),0.5))

    print(d)