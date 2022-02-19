#!/usr/bin/env python
"""
Author: Nima Rezaeian
"""

import numpy as np
import sympy as sp

np.set_printoptions(precision=4,suppress=True)

#-----------------------------3D Functions--------------------------------------

def skew_3d(omega):
    """
    Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.
    
    Args:
    omega - (3,1) sympy Matrix: the rotation matrix
    
    Returns:
    omega_hat - (3,3) sympy Matrix: the corresponding skew symmetric matrix
    """

    skew = sp.Matrix([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]]
                    )
    return skew


def skew_to_rot_3d(skew):
    """
    Converts a skew symmetric matrix representing a rotation axis back to a rotation matrix.

    Args:
    skew - (3,3) sympy Matrix: the skew symmetric matrix

    Returns:
    omega - (3, 1) sympy Matrix: the rotation matrix
    """
    return sp.Matrix([[-skew[1,2]], [skew[0,2]], [-skew[0,1]]])


def rotation_3d(omega, theta):
    """
    Computes a 3D rotation matrix given a rotation axis and angle of rotation.
    
    Args:
    omega - (3,1) sympy Matrix: the axis of rotation
    theta: the angle of rotation
    
    Returns:
    rot - (3,3) sympy Matrix: the resulting rotation matrix
    """

    mag = (omega.T * omega)[0]
    skew = skew_3d(omega)
    skew2 = skew * skew
    R = sp.eye(3) + (skew / mag) * sp.sin(mag * theta) \
                                    + (skew2 / (mag ** 2)) * (1 - sp.cos(mag * theta))
    return R


def hat_3d(xi):
    """
    Converts a 3D twist to its corresponding 4x4 matrix representation
    
    Args:
    xi - (6,1) sympy Matrix: the 3D twist
    
    Returns:
    xi_hat - (4,4) sympy Matrix: the corresponding 4x4 matrix
    """

    v = xi[0:3, :]
    w_hat = skew_3d(xi[3:6])
    return sp.Matrix.vstack(sp.Matrix.hstack(w_hat, v), sp.zeros(1, 4))


def homog_to_twist_3d(homog):
    """
    Converts a homog representing a twist axis back to a twist matrix.

    Args:
    homog - (4,4) sympy Matrix: the homog transform representing a twist

    Returns:
    twist - (6, 1) sympy Matrix: the twist matrix
    """
    v = homog[0:3, 3]
    w = skew_to_rot_3d(homog[0:3, 0:3]) 
    return sp.Matrix.vstack(v, w)


def homog_3d(xi, theta):
    """
    Computes a 4x4 homogeneous transformation matrix given a 3D twist and a 
    joint displacement.
    
    Args:
    xi - (6,1) sympy Matrix: the 3D twist
    theta - Symbol: the joint displacement
    Returns:
    g - (4,4) sympy Matrix: the resulting homogeneous transformation matrix
    """

    w = xi[3:6, :]
    norm_w = (w.T * w)[0]
    v = xi[0:3, :]
    if norm_w == 0:
        top = sp.Matrix.hstack(sp.eye(3), theta * v)
        bottom = sp.Matrix([[0, 0, 0, 1]])
        return sp.Matrix.vstack(top, bottom)
    else:
        top_left = rotation_3d(w, theta)
        top_right = (sp.eye(3) - top_left) * (skew_3d(w) * v)
        top_right += (w * w.T * v) * theta
        top_right = top_right / (norm_w ** 2)
        top = sp.Matrix.hstack(top_left, top_right)
        bottom = sp.Matrix([[0, 0, 0, 1]])
        return sp.Matrix.vstack(top, bottom)


def prod_exp(xi, theta):
    """
    Computes the product of exponentials for a kinematic chain, given 
    the twists and displacements for each joint.
    
    Args:
    xi - list of sympy Matrices: the twists for each joint
    theta - list of symbols: the displacement of each joint
    
    Returns:
    g - (4,4) sympy Matrix: the resulting homogeneous transformation matrix
    """

    N = len(xi)
    prod = sp.eye(4)
    for i in range(N):
        prod = prod * homog_3d(xi[i], theta[i])
    return prod
    

def adjoint(homog):
    """
    Find the adjoint transformation of the given homogenous matrix.

    Args: homog - (4,4) sympy Matrix.

    Returns: adjoint - (6, 6) sympy Matrix
    """

    p = homog[0:3, 3]
    R = homog[0:3, 0:3]
    up_right = skew_3d(p) * R
    up = sp.Matrix.hstack(R, up_right)
    down = sp.Matrix.hstack(sp.zeros(3, 3), R)
    return sp.Matrix.vstack(up, down)


def adjoint_inv(homog):
    """
    Find the adjoint transformation of the given homogenous matrix.

    Args: homog - (4,4) sympy Matrix.

    Returns: adjoint - (6, 6) sympy Matrix
    """

    p = homog[0:3, 3]
    R = homog[0:3, 0:3]
    up_right = -R.T * skew_3d(p)
    up = sp.Matrix.hstack(R.T, up_right)
    down = sp.Matrix.hstack(sp.zeros(3, 3), R.T)
    return sp.Matrix.vstack(up, down)


def spatial_jacobian(xi, theta):
    """
    Computes the spatial Jacobian given a list of twists and list of angles
    """
    N_dof = len(xi)
    J_spatial = sp.zeros(6, N_dof)
    cum_exp = sp.eye(4)
    for i in range(N_dof):
        J_spatial[:, i] = adjoint(cum_exp) * xi[i]
        cum_exp = cum_exp * homog_3d(xi[i], theta[i])
    return J_spatial

def body_jacobian(xi, theta, gst_0):
    N_dof = len(xi)
    J_body = sp.zeros(6, N_dof)
    tail_exp = [homog_3d(xi[-1], theta[-1]) * gst_0]
    for i in range(N_dof-2, -1, -1):
        tail_exp.append(tail_exp[-1] * homog_3d(xi[i], theta[i]))
    tail_exp.reverse()
    
    for i in range(N_dof):
        J_body[:, i] = adjoint_inv(tail_exp[i] * gst_0) * xi[i]
    return J_body