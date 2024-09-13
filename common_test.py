import unittest
from common import dicts_to_csv

class TestDictsToCsv(unittest.TestCase):
    def test_basic_case(self):
        dict_list = [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25}
        ]
        expected_csv = "age,name\r\n30,Alice\r\n25,Bob\r\n"
        result = dicts_to_csv(dict_list)
        self.assertEqual(result, expected_csv)

    def test_missing_fields(self):
        dict_list = [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob'},  # Missing 'age'
            {'age': 22}  # Missing 'name'
        ]
        expected_csv = "age,name\r\n30,Alice\r\n,Bob\r\n22,\r\n"
        result = dicts_to_csv(dict_list)
        self.assertEqual(result, expected_csv)

    def test_list_values(self):
        dict_list = [
            {'name': 'Alice', 'hobbies': ['reading', 'chess']},
            {'name': 'Bob', 'hobbies': ['guitar', 'football']}
        ]
        expected_csv = "hobbies,name\r\nreading chess,Alice\r\nguitar football,Bob\r\n"
        result = dicts_to_csv(dict_list)
        self.assertEqual(result, expected_csv)

    def test_mixed_data_types(self):
        dict_list = [
            {'name': 'Alice', 'age': 30, 'hobbies': ['reading', 'chess']},
            {'name': 'Bob', 'age': 25, 'hobbies': 'guitar'}
        ]
        expected_csv = "age,hobbies,name\r\n30,reading chess,Alice\r\n25,guitar,Bob\r\n"
        result = dicts_to_csv(dict_list)
        self.assertEqual(result, expected_csv)

    def test_empty_dict(self):
        dict_list = [{}]  # A dict with no data
        expected_csv = "\r\n"
        result = dicts_to_csv(dict_list)
        self.assertEqual(result, expected_csv)

    def test_no_dicts(self):
        dict_list = []  # An empty list of dicts
        expected_csv = ""
        result = dicts_to_csv(dict_list)
        self.assertEqual(result, expected_csv)

if __name__ == '__main__':
    unittest.main()
