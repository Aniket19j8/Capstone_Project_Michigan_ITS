from typer.testing import CliRunner

from its_intake.cli import app


runner = CliRunner()


def test_help_shows_run_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "run-e2e" in result.stdout
    assert "resolve-ticket" in result.stdout


def test_root_invocation_missing_audio_file() -> None:
    result = runner.invoke(app, ["--audio", "does_not_exist.wav"])
    assert result.exit_code != 0
    assert "Audio file not found: does_not_exist.wav" in result.output


def test_run_subcommand_missing_audio_file() -> None:
    result = runner.invoke(app, ["run", "--audio", "does_not_exist.wav"])
    assert result.exit_code != 0
    assert "Audio file not found: does_not_exist.wav" in result.output


def test_resolve_ticket_missing_file() -> None:
    result = runner.invoke(app, ["resolve-ticket", "--ticket-json", "missing_ticket.json"])
    assert result.exit_code != 0
    assert "Ticket JSON not found: missing_ticket.json" in result.output
