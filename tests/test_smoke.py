import agentos


def test_version() -> None:
    assert agentos.__version__ == "0.1.0"


def test_main_runs(capsys) -> None:
    agentos.main()
    assert "agentos" in capsys.readouterr().out
