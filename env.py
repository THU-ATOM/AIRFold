import argparse
import os
from collections import OrderedDict
from pathlib import Path

DEFAULT_ENV_PATH = ".env"


class Env:
    def __init__(self, path=".env"):
        self.path = Path(path)
        self.lines, self.variables = self._parse_env()

    def _parse_env(self):
        lines = []
        variables = OrderedDict()
        if self.path.is_file():
            with open(self.path) as f:
                for i, line in enumerate(f):
                    lines.append(line)
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, val = line.split("=")
                    variables[key] = {"line": i, "key": key, "val": val}
        return lines, variables

    def __repr__(self):
        _strs = []
        for key, val in self.variables.items():
            _strs.append(f"  {key}: {val['val']}")
        return "\n".join(_strs)

    def __getitem__(self, key):
        return self.variables[key]["val"]

    def __setitem__(self, key, val):
        if key in self.variables:
            self.variables[key]["val"] = val
        else:
            self.variables[key] = {
                "line": len(self.lines),
                "key": key,
                "val": val,
            }
        self._update_line(self.variables[key])

    def __contains__(self, key):
        return key in self.variables

    def _update_line(self, var):
        line = f"{var['key']}={var['val']}\n"
        if var["line"] < len(self.lines):
            self.lines[var["line"]] = line
        else:
            self.lines.append(line)

    def save(self):
        with open(self.path, "w") as f:
            f.writelines(self.lines)


def main():
    parser = argparse.ArgumentParser(
        description="The core script of experiment management."
    )
    parser.add_argument("action", nargs="?", default="enter")

    args = parser.parse_args()

    _set_env(verbose=(args.action == "start" or args.action == "startd"))


def _set_env(env_path=DEFAULT_ENV_PATH, verbose=False):
    e = Env(env_path)
    e["UID"] = os.getuid()
    e["GID"] = os.getgid()
    e["USER_NAME"] = os.getlogin()


    e.save()

    if verbose:
        print(f"Your setting ({env_path}):\n{e}\n")
    return e


if __name__ == "__main__":
    main()
