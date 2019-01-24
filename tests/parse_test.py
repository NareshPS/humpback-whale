#Unittest
import unittest as ut

#Parse string to its typed value
from common.parse import kv_str_to_tuple

class TestFunctions(ut.TestCase):
    def kv_str_to_tuple_verify(self, value, result):
        kv_tuple = kv_str_to_tuple(value)

        #Assert
        self.assertSequenceEqual(kv_tuple, result)

    def test_kv_str_to_tuple_valid_values(self):
        #Integer values
        self.kv_str_to_tuple_verify("key=10", ('key', 10))

        #Float values
        self.kv_str_to_tuple_verify("key=5.3", ('key', 5.3))

        #Boolean values
        self.kv_str_to_tuple_verify("key=true", ('key', True))

        #String values
        self.kv_str_to_tuple_verify("key=hello", ('key', 'hello'))

    def test_kv_str_to_tuple_invalid_values(self):
        with self.assertRaises(ValueError):
            kv_str_to_tuple('key:value')