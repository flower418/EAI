import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Unnormalized quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        Normalized quaternion with shape (4,)
    """
    # normalize 的时候要除以 L2 范数
    return q / np.linalg.norm(q)

def quat_conjugate(q: np.ndarray) -> np.ndarray: # conjugate：共轭
    """
    Return the conjugate of the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The conjugate of the quaternion with shape (4,)
    """
    # order: wxyz
    return q * [1, -1, -1, -1] # vectorization

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    np.ndarray
        The multiplication result with shape (4,)
    """
    # 直接展开
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + w2 * x1 + y1 * z2 - y2 * z1
    y = w1 * y2 + w2 * y1 - x1 * z2 + z1 * x2
    z = w1 * z2 + w2 * z1 + x1 * y2 - x2 * y1

    return np.array([w, x, y, z])

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Use quaternion to rotate a 3D vector.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)
    v: np.ndarray
        Vector with shape (3,)

    Returns
    -------
    np.ndarray
        The rotated vector with shape (3,)
    """
    v1 = np.hstack((0, v)) # hstack: horizontal stack
    q_i = quat_conjugate(q)
    temp_ans = quat_multiply(quat_multiply(q, v1), q_i)
    return temp_ans[1:]

def quat_relative_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute the relative rotation angle between the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    float
        The relative rotation angle in radians, greater than or equal to 0.
    """
    theta = 2 * np.arccos(np.abs(np.sum(q1 * q2)))
    return theta

def interpolate_quat(q1: np.ndarray, q2: np.ndarray, ratio: float) -> np.ndarray:
    """
    Interpolate between two quaternions with given ratio.

    Please use Spherical linear interpolation (SLERP) here.

    When the ratio is 0, return q1; when the ratio is 1, return q2.

    The interpolation should be done in the shortest minor arc connecting the quaternions on the unit sphere.

    If there are multiple correct answers, you can output any of them.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)
    ratio: float
        The ratio of interpolation, should be in [0, 1]

    Returns
    -------
    np.ndarray
        The interpolated quaternion with shape (4,)

    Note
    ----
    What should be done if the inner product of the quaternions is negative?
    """
    # 当 q1,q2 点积为负时，需要将一个取反，在小弧上进行插值
    # 不能直接取绝对值，因为插值用到的 q1,q2会改变，不能只改变角度而不改变真正用于插值的 quaternion
    dot = np.sum(q1 * q2)
    if (dot < 0):
        q1 = -q1
    
    theta = np.arccos(np.sum(q1 * q2))
    qt = (q1 * np.sin((1 - ratio) * theta) + q2 * np.sin(ratio * theta)) / np.sin(theta)
    return qt

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to rotation matrix.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    ret = np.zeros((3, 3))
    w, x, y, z = q

    ret[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    ret[0, 1] = 2 * (x * y - w * z)
    ret[0, 2] = 2 * (x * z + y * w)
    ret[1, 0] = 2 * (x * y + w * z)
    ret[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    ret[1, 2] = 2 * (y * z - w * x)
    ret[2, 0] = 2 * (x * z - w * y)
    ret[2, 1] = 2 * (y * z + w * x)
    ret[2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return ret

def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to quaternion.

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    q = np.zeros(4)

    tr = mat[0, 0] + mat[1, 1] + mat[2, 2]
    w = np.sqrt(tr + 1) / 2
    x = (mat[2, 1] - mat[1, 2]) / (4 * w)
    y = (mat[0, 2] - mat[2, 0]) / (4 * w)
    z = (mat[1, 0] - mat[0, 1]) / (4 * w)

    q[0], q[1], q[2], q[3] = w, x, y, z
    return q

def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi.

    If there are multiple answers, you can output any.

    Parameters
    ----------
    q: np.ndarray
        The quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    ret = np.zeros(3)

    w = q[0]
    v = q[1:]
    theta = 2 * np.arccos(w)

    # 注意这里要考虑，当 theta 大于 pi 时，要将其限制在 0-pi,故要取负
    if theta > np.pi:
        v = -v
        theta = 2 * np.pi - theta
    
    if theta < 1e-12: # =0
        return ret
    return theta / np.sin(theta / 2) * v

def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to quaternion.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    theta = np.linalg.norm(aa)
    v = np.sin(theta / 2) / theta * aa
    w = np.cos(theta / 2)

    q = np.zeros(4)
    q[0] = w
    q[1:] = v
    return q

def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to rotation matrix.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    q = axis_angle_to_quat(aa)
    mat = quat_to_mat(q)
    return mat

def mat_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    q = mat_to_quat(mat)
    axis = quat_to_axis_angle(q)
    return axis

def uniform_random_quat() -> np.ndarray:
    """
    Generate a random quaternion with uniform distribution.

    Returns
    -------
    np.ndarray
        The random quaternion with shape (4,)
    """
    raise NotImplementedError("Implement this function")


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """
    Convert roll-pitch-yaw euler angles into rotation matrix.

    This is required since URDF use this as rotation representation.

    Parameters
    ----------
    rpy: np.ndarray
        The euler angles with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    roll, pitch, yaw = rpy

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x  # Matrix multiplication in ZYX order
    return R
