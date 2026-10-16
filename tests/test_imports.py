"""The core package imports without any optional extra installed or loaded."""

import subprocess
import sys


def describe_core_package():
    def it_imports_without_numpy_torch_mlx_or_boto3():
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import gradientmail, sys; "
                "leaked = {'numpy', 'torch', 'mlx', 'boto3', 'zstandard'} & sys.modules.keys(); "
                "assert not leaked, leaked",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
