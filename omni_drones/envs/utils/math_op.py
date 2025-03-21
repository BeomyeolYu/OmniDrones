import torch

def hat(x):
    ensure_vector(x, 3)

    hat_x = torch.tensor([[0.0, -x[2], x[1]], \
                          [x[2], 0.0, -x[0]], \
                          [-x[1], x[0], 0.0]], device=x.device)
                    
    return hat_x


def vee(M):
    """Returns the vee map of a given 3x3 matrix.
    Args:
        x: (3x3 numpy array) hat of the input vector
    Returns:
        (3x1 numpy array) vee map of the input matrix
    """
    ensure_skew(M, 3)

    vee_M = torch.tensor([M[2,1], M[0,2], M[1,0]], device=M.device)

    return vee_M


def deriv_unit_vector(A, A_dot, A_2dot):
    """Returns the unit vector and it's derivatives for a given vector.
    Args:
        A: (3x1 numpy array) vector
        A_dot: (3x1 numpy array) first derivative of the vector
        A_2dot: (3x1 numpy array) second derivative of the vector
    Returns:
        q: (3x1 numpy array) unit vector of A
        q_dot: (3x1 numpy array) first derivative of q
        q_2dot: (3x1 numpy array) second derivative of q
    """

    ensure_vector(A, 3)
    ensure_vector(A_dot, 3)
    ensure_vector(A_2dot, 3)

    nA = torch.norm(A)

    if abs(torch.norm(nA)) < 1.0e-9:
        raise ZeroDivisionError('The 2-norm of A should not be zero')

    nA3 = nA * nA * nA
    nA5 = nA3 * nA * nA

    A_A_dot = A.dot(A_dot)

    q = A / nA
    q_dot = A_dot / nA \
        - A.dot(A_A_dot) / nA3

    q_2dot = A_2dot / nA \
        - A_dot.dot(2.0 * A_A_dot) / nA3 \
        - A.dot(A_dot.dot(A_dot) + A.dot(A_2dot)) / nA3 \
        + 3.0 * A.dot(A_A_dot).dot(A_A_dot)  / nA5

    return (q, q_dot, q_2dot)


def ensure_vector(x, n):
    """Make sure the given input array x is a vector of size n.
    Args:
        x: (nx1 numpy array) vector
        n: (int) desired length of the vector
    Returns:
        True if the input array is satisfied with size constraint. Raises an
        exception otherwise.
    """

    x = torch.atleast_2d(x)  # Make sure the array is atleast 2D.

    if not len(x.ravel()) == n:
        raise ValueError(f'Input array must be of length {n}, but detected {x.shape}')
    
    return True


def ensure_matrix(x, m, n):
    """Make sure the given input array x is a matrix of size mxn.
    Args:
        x: (mxn numpy array) array
        m: (int) desired number of rows
        n: (int) desired number of columns
    Returns:
        True if the input array is satisfied with size constraint. Raises an
        exception otherwise.
    """

    x = torch.atleast_2d(x)  # Make sure the array is atleast 2D.

    if x.shape != (m, n):
        raise ValueError(f'Input array must be of size {m}x{n}, but detected {x.shape}')
    
    return True


def ensure_skew(x, n):
    """Make sure the given input array is a skew-symmetric matrix of size nxn.
    Args:
        x: (nxn numpy array) array
        m: (int) desired number of rows and columns
    Returns:
        True if the input array is a skew-symmetric matrix. Raises an
        exception otherwise.
    """
    ensure_matrix(x, n, n)
    
    if not torch.allclose(x.T, -x):
        raise ValueError('Input array must be a skew-symmetric matrix')
    
    return True

class IntegralErrorVec3:
    def __init__(self, device):
        self.device = device
        self.error = torch.zeros(3, device=self.device)
        self.integrand = torch.zeros(3, device=self.device)

    def integrate(self, current_integrand, dt):
        self.error += (self.integrand + current_integrand) * dt / 2.0
        self.integrand = current_integrand

    def set_zero(self):
        self.error.zero_()
        self.integrand.zero_()


class IntegralError:
    def __init__(self, device):
        self.device = device
        self.error = torch.tensor(0.0, device=self.device)
        self.integrand = torch.tensor(0.0, device=self.device)

    def integrate(self, current_integrand, dt):
        self.error += (self.integrand + current_integrand) * dt / 2.0
        self.integrand = current_integrand

    def set_zero(self):
        self.error.zero_()
        self.integrand.zero_()


def ensure_SO3(R, tolerance=1e-6):
    """ Make sure the given input array is in SO(3).

    Args:
        x: (3x3 numpy array) matrix
        tolerance: Tolerance level for considering the magnitude as 1

    Returns:
        True if the input array is in SO(3). Raises an exception otherwise.
    """
    # Calculate the magnitude (norm) of the matrix
    magnitude = torch.det(R)

    # R matrix should satisfy R^T@R = I and det(R) = 1:
    if torch.allclose(R.T @ R, torch.eye(3, device=R.device), rtol=tolerance) and torch.isclose(magnitude, torch.tensor(1.0, device=R.device), rtol=tolerance):
        return R
    else: 
        U, s, VT = psvd(R)
        R = U @ VT.T # Re-orthonormalized R
        return R
    
def psvd(A):
    assert A.shape == (3, 3)
    U, s, VT = torch.linalg.svd(A)
    detU = torch.det(U)
    detV = torch.det(VT)
    U[:, 2] *= detU
    VT[2, :] *= detV
    s[2] *= detU * detV
    return U, s, VT.T

def q_to_R(q):
    """Converts a quaternion of a rotation matrix in SO(3).

    Args:
        q: (4x1 numpy array) quaternion in [x, y, z, w] format

    Returns:
        R: (3x3 numpy array) rotation matrix corresponding to the quaternion
    """

    # TODO: ensure quaternion instead of ensure vector
    ensure_vector(q, 4)

    R = torch.eye(3, device=q.device)
    q13 = q[:3]
    q4 = q[3]
    hat_q = hat(q13)
    R += 2.0 * q4 * hat_q + 2.0 * hat_q @ hat_q
    return R

def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    matrix = torch.tensor([
        [1 - (tyy + tzz), txy - twz, txz + twy],
        [txy + twz, 1 - (txx + tzz), tyz - twx],
        [txz - twy, tyz + twx, 1 - (txx + tyy)]
    ], device=quaternion.device)

    return matrix

def quat_rotate_inverse(q, v):
    """ Rotates a vector v using the inverse of quaternion q.
    
    Args:
        q: (N, 4) tensor representing a batch of quaternions [w, x, y, z]
        v: (N, 3) tensor representing a batch of vectors

    Returns:
        Rotated vectors as a (N, 3) torch tensor
    """
    q_w = q[:, 0].unsqueeze(-1)  # (N, 1)
    q_vec = q[:, 1:]  # (N, 3)

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * (2.0 * q_w)
    c = q_vec * torch.bmm(q_vec.unsqueeze(1), v.unsqueeze(2)).squeeze(-1) * 2.0

    return a - b + c


# Normalization state vectors: [max, min] -> [-1, 1]
def state_normalization(state, x_lim, v_lim, W_lim):
    x, v, R_vec, W = state[0], state[1], state[2], state[3]
    x_norm, v_norm, W_norm = x/x_lim, v/v_lim, W/W_lim
    '''
    R = ensure_SO3(R_vec.reshape(3, 3, order='F')) # re-orthonormalization if needed
    
    R_vec = R.reshape(9, 1, order='F').flatten()
    return x_norm, v_norm, R_vec, W_norm
    '''

    return x_norm, v_norm, R_vec, W_norm

class IntegralErrorVec3:
    def __init__(self, num_envs, device):
        self.error = torch.zeros((num_envs, 1, 3), device=device)
        self.integrand = torch.zeros((num_envs, 1, 3), device=device)

    def integrate(self, current_integrand: torch.Tensor, dt: float):
        # current_integrand: (num_envs, 1, 3)
        self.error += 0.5 * (self.integrand + current_integrand) * dt
        self.integrand = current_integrand

    def set_zero(self, env_ids=None):
        if env_ids is None:
            self.error.zero_()
            self.integrand.zero_()
        else:
            self.error[env_ids] = 0.
            self.integrand[env_ids] = 0.


class IntegralError:
    def __init__(self, num_envs, device):
        self.error = torch.zeros((num_envs, 1, 1), device=device)
        self.integrand = torch.zeros((num_envs, 1, 1), device=device)

    def integrate(self, current_integrand: torch.Tensor, dt: float):
        # current_integrand: (num_envs, 1, 1)
        self.error += 0.5 * (self.integrand + current_integrand) * dt
        self.integrand = current_integrand

    def set_zero(self, env_ids=None):
        if env_ids is None:
            self.error.zero_()
            self.integrand.zero_()
        else:
            self.error[env_ids] = 0.
            self.integrand[env_ids] = 0.
