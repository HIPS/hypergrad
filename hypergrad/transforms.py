import numpy as np

def translate(Lx, Ly, nx, ny):
    x_transform = np.diag(np.ones(Lx - abs(nx)), -nx).reshape([1, Lx, 1, Lx])
    y_transform = np.diag(np.ones(Ly - abs(ny)), -ny).reshape([Ly, 1, Ly, 1])
    return (x_transform * y_transform).reshape([Lx * Ly, Lx * Ly])
    
def matrix_exp(A, t):
    D, V = np.linalg.eig(A)
    return V.dot(np.diag(np.exp(t * D)).dot(V.T))
