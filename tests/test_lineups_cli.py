import builtins
import importlib
import sys
import unittest
from unittest import mock

from pipeline import lineups


def _import_raising_bs4_missing(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pipeline.common.lineups.ingest":
        raise ModuleNotFoundError("No module named 'bs4'", name="bs4")
    return _import_raising_bs4_missing.original(name, globals, locals, fromlist, level)


_import_raising_bs4_missing.original = builtins.__import__


class LineupsCliTests(unittest.TestCase):
    def test_lineups_main_fail_soft_on_missing_bs4(self):
        with mock.patch("pipeline.lineups.load_dotenv"), \
             mock.patch("builtins.__import__", side_effect=_import_raising_bs4_missing), \
             mock.patch("builtins.print") as print_mock:
            rc = lineups.main([])

        self.assertEqual(rc, 0)
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("dependencies are missing", printed)
        self.assertIn("Fail-soft mode enabled", printed)

    def test_lineups_main_strict_on_missing_bs4(self):
        with mock.patch("pipeline.lineups.load_dotenv"), \
             mock.patch("builtins.__import__", side_effect=_import_raising_bs4_missing), \
             mock.patch("builtins.print"):
            rc = lineups.main(["--strict"])

        self.assertEqual(rc, 1)

    def test_lineups_package_import_does_not_require_ingest_deps(self):
        with mock.patch("builtins.__import__", side_effect=_import_raising_bs4_missing):
            sys.modules.pop("pipeline.common.lineups", None)
            module = importlib.import_module("pipeline.common.lineups")

        self.assertTrue(hasattr(module, "build_lineup_match_features"))
        self.assertTrue(hasattr(module, "load_lineup_entries"))


if __name__ == "__main__":
    unittest.main()
