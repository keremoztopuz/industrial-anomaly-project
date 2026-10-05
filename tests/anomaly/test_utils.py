import json
import tempfile
import unittest
from pathlib import Path

from anomaly.utils import write_json


class WriteJsonTests(unittest.TestCase):
    def test_writes_sorted_json_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            write_json(path, {"b": 1, "a": 2})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "a": 2,\n  "b": 1\n}\n')
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 2, "b": 1})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
