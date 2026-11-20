import unittest

import pyarrow as pa

from petlm.fineweb_parquet import read_row_groups_resilient


class Handle:
    def __init__(self, closed):
        self.closed = closed

    def close(self):
        self.closed.append(True)


class Reader:
    def __init__(self, failures):
        self.failures = failures

    def read_row_groups(self, groups, columns):
        key = tuple(groups)
        if self.failures.get(key, 0):
            self.failures[key] -= 1
            raise TimeoutError('transient range timeout')
        return pa.table({'group': groups})


class FineWebParquetTests(unittest.TestCase):
    def test_chunk_retry_reopens_and_closes_handles(self):
        failures = {(1, 2): 1}
        closed = []
        sleeps = []
        opened_urls = []

        def opener(url):
            opened_urls.append(url)
            return Handle(closed), Reader(failures)

        table = read_row_groups_resilient(
            'https://example.test/data.parquet', [1, 2, 3], ['group'],
            chunk_size=2, attempts=3, opener=opener, sleeper=sleeps.append)

        self.assertEqual(table.column('group').to_pylist(), [1, 2, 3])
        self.assertEqual(len(closed), 3)
        self.assertEqual(sleeps, [2])
        self.assertEqual(opened_urls, ['https://example.test/data.parquet'] * 3)

    def test_invalid_retry_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            read_row_groups_resilient('url', [1], ['text'], chunk_size=0)


if __name__ == '__main__':
    unittest.main()
