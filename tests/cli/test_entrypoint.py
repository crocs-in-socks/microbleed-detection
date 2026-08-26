from unittest.mock import Mock

from typer.testing import CliRunner

from microbleednet.cli import entrypoint

app = entrypoint.app

runner = CliRunner()


def test_verbose_and_quiet_are_mutually_exclusive() -> None:
    result = runner.invoke(app, ["--verbose", "--quiet", "index-data", "--help"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_verbose_flag_is_accepted() -> None:
    result = runner.invoke(app, ["--verbose", "index-data", "--help"])
    assert result.exit_code == 0, result.output


def test_quiet_flag_is_accepted() -> None:
    result = runner.invoke(app, ["--quiet", "index-data", "--help"])
    assert result.exit_code == 0, result.output


def test_main_runs_the_cli(monkeypatch) -> None:
    cli = Mock()
    monkeypatch.setattr(entrypoint, "app", cli)

    entrypoint.main()

    cli.assert_called_once_with()
