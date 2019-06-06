import unittest
import torch

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test(self):
        tensor = torch.Tensor([1., 2., 3., 4.])
        a, b = tensor.topk(2)
        print(a)
        print(b)


if __name__ == '__main__':
    unittest.main()
