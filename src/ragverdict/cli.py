"""Click CLI entry point: `ragverdict run <config.yaml>`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ragverdict import __version__
from ragverdict.adapters.loader import AdapterLoadError
from ragverdict.config import ConfigError
from ragverdict.runner import Runner, RunnerError


@click.group(name="ragverdict")
@click.version_option(__version__, prog_name="ragverdict")
def cli() -> None:
    """ragverdict — pytest for RAG agents."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("./report"),
    show_default=True,
    help="Directory to write report.json and report.md.",
)
@click.option(
    "--no-judge",
    is_flag=True,
    default=False,
    help="Skip LLM-as-judge entirely. Hard assertions still run; WEAK verdicts and "
    "citation support scoring are disabled.",
)
def run(config_path: Path, out_dir: Path, no_judge: bool) -> None:
    """Run the evaluation defined in CONFIG_PATH."""
    try:
        runner = Runner.from_config_path(
            config_path,
            out_dir=out_dir,
            judge=None if no_judge else "auto",
        )
        _results, exit_code = runner.execute()
    except (ConfigError, AdapterLoadError, RunnerError) as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)
    sys.exit(exit_code)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
