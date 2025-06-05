"""
Singular Vector Decomposition :
Used to calculate pseudoinverse when matrix is Singular or rectangle.
Fcatorize mat A(mxn) as  = U.Z.Vt
U,V are orthogonal matrices
U = cols are left singular vectors of A = (mxn)
V = toes are right singular vectors of A = (nxn)
Z = diagonal matrix of non nega values of A: singular valyes of A

Singular values oF A are: eigen valyes of At.A or A.At

Steps:
1. compute At.A and find its eigen values(gives square of singular values)  and eignen vectors (makes cols of V)
2. Take sqrt of eigen values of At.A, place them as dialogal values of Z

Substeps:
1.1 eigen vectors of At.A - to get V - eqn => (At.A - eigenvalue1)v = 0 -> gives v1
                                            => (At.A - eigenvalue2)v = 0 -> gives v2
                                            v1,v2 = are column vectrs
                                            V = [v1 v2]

1.2 find U : Av1 = eigenvalue1.u1 and Av2 = eigenvalue2.u2
            U = [u1 u2] ; u1,u2 are column vectrs


1.3 find Z :
[eigenvalue1 0  : sqrt(eigenvalues) at diagonals
0 eigenvalue2]

"""