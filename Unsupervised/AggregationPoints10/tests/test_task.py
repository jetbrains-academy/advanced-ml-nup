import unittest
import pandas as pd

from task import majority_voting


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'object': [1, 1, 2, 2, 3, 3, 3],
            'user': ['u1', 'u2', 'u1', 'u2', 'u1', 'u2', 'u1'],
            'label': ['A', 'A', 'B', 'B', 'A', 'B', 'B']
        })
        self.golden_data = pd.DataFrame({
            'object': [1, 2, 3],
            'label': ['A', 'B', 'A']
        })
        self.golden_data.set_index('object', inplace=True)

    def test_majority_voting(self):
        result = majority_voting(self.data)
        expected_result = pd.Series(
                    index=pd.Index(data=[1, 2, 3], name='object'),
                    data=['A', 'B', 'B']
                )
        pd.testing.assert_series_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main()
