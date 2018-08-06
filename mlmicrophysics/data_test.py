from .data import load_cam_output
import unittest


class TestData(unittest.TestCase):
    def setUp(self):
        self.bad_path = "/blah/de/blah"
        return

    def test_load_cam_output(self):
        with self.assertRaises(FileNotFoundError):
            load_cam_output(self.bad_path)

