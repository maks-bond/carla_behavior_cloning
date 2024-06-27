import math

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Multiplication is only supported between two Quaternions")

    def norm(self):
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        n = self.norm()
        self.w /= n
        self.x /= n
        self.y /= n
        self.z /= n

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        n = self.norm()
        return self.conjugate() * (1.0 / (n * n))

    # Looks like this is equal to rotating around X (roll) first, then Y (pitch), then Z (yaw).
    def rotate_point(self, point):
        """
        Rotate a 3D point using this quaternion.
        :param point: A tuple (x, y, z) representing the point.
        :return: A tuple (x', y', z') representing the rotated point.
        """
        p = Quaternion(0, *point)
        q_conj = self.conjugate()
        rotated_p = self * p * q_conj
        return (rotated_p.x, rotated_p.y, rotated_p.z)

    def inverse_rotate_point(self, point):
        """
        Rotate a 3D point using the inverse of this quaternion.
        :param point: A tuple (x, y, z) representing the point.
        :return: A tuple (x', y', z') representing the rotated point.
        """
        inverse_q = self.inverse()
        return inverse_q.rotate_point(point)
    
    @staticmethod
    def from_yzx_euler(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cz = math.cos(pitch * 0.5)
        sz = math.sin(pitch * 0.5)
        cx = math.cos(roll * 0.5)
        sx = math.sin(roll * 0.5)

        qw = cy * cz * cx + sy * sz * sx
        qx = cy * cz * sx - sy * sz * cx
        qy = cy * sz * cx + sy * cz * sx
        qz = sy * cz * cx - cy * sz * sx

        return Quaternion(qw, qx, qy, qz)

    @staticmethod
    def from_xyz_euler(roll, pitch, yaw):
        """
        Create a quaternion from Euler angles.
        :param pitch: Rotation around x-axis in radians
        :param yaw: Rotation around y-axis in radians
        :param roll: Rotation around z-axis in radians
        :return: Quaternion
        """
        cy = math.cos(roll * 0.5)
        sy = math.sin(roll * 0.5)
        cp = math.cos(yaw * 0.5)
        sp = math.sin(yaw * 0.5)
        cr = math.cos(pitch * 0.5)
        sr = math.sin(pitch * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(w, x, y, z)