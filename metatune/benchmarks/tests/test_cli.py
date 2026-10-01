from metatune.benchmarks.cli import build_parser


def test_cli_parser_has_suite_command():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    assert "suite" in choices
    assert "run" in choices
    assert "profile" in choices
