"""CLI entry-point exposing handy Hugging Face automation commands."""
from __future__ import annotations

import typer

from hfhub.auth import get_default_namespace, logged_in_user
from hfhub.io import push_local_folder_as_dataset, push_local_folder_as_space

cli_app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@cli_app.command()
def whoami(url: bool = typer.Option(default=True)):
    """Print the currently-logged-in HF account."""
    logged_in_user(print_url=url)


@cli_app.command()
def ns(namespace: str = typer.Option(None, "--value", "-v")):
    """Show (or compute) the default organisation/user namespace."""
    val = namespace or get_default_namespace()
    typer.echo(val)


@cli_app.command()
def push_data(root: str, repo: str, private: bool = True):
    """Upload a local folder as a brand-new dataset on the Hub."""
    push_local_folder_as_dataset(local_root=root, ds_id=repo, private=private)


@cli_app.command()
def deploy(app_root: str, space: str, sdk: str = "gradio"):
    """Deploy a local gradio/static app as a live Space."""
    push_local_folder_as_space(local_root=app_root, sp_id=space, sdk=sdk)


if __name__ == "__main__":
    cli_app()
