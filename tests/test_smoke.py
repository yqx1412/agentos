import pytest

import agentos
from agentos.cli import main


def test_version() -> None:
    assert agentos.__version__ == "0.1.0"


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "agentos 0.1.0" in capsys.readouterr().out
