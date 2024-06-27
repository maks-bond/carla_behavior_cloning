import math
import unittest
from quaternion import Quaternion

class TestQuaternion(unittest.TestCase):

    def test_quaternion_multiplication(self):
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(2, -1, 1, -2)
        q3 = q1 * q2
        self.assertAlmostEqual(q3.w, 9)
        self.assertAlmostEqual(q3.x, -7)
        self.assertAlmostEqual(q3.y, 7)
        self.assertAlmostEqual(q3.z, 11)

    def test_quaternion_xyz_rotation(self):
        q = Quaternion.from_xyz_euler(math.radians(45), math.radians(30), math.radians(60))
        point = (1, 0, 0)
        rotated_point = q.rotate_point(point)
        expected_rotated_point = (0.43301, 0.750, -0.5)  # Calculated from quaternion rotation
        for r, e in zip(rotated_point, expected_rotated_point):
            self.assertAlmostEqual(r, e, places=5)

    def test_quaternion_yzx_rotation(self):
        q = Quaternion.from_yzx_euler(math.radians(45), math.radians(30), math.radians(60))
        point = (1, 0, 0)
        rotated_point = q.rotate_point(point)
        expected_rotated_point = (0.43301, 0.88388, 0.17678)  # Calculated from quaternion rotation
        for r, e in zip(rotated_point, expected_rotated_point):
            self.assertAlmostEqual(r, e, places=5)

    def test_quaternion_inverse_rotation(self):
        q = Quaternion.from_yzx_euler(math.radians(45), math.radians(30), math.radians(60))
        point = (1, 0, 0)
        rotated_point = q.rotate_point(point)
        q_inv = q.inverse()
        restored_point = q_inv.rotate_point(rotated_point)
        for r, e in zip(restored_point, point):
            self.assertAlmostEqual(r, e)

if __name__ == '__main__':
    unittest.main()