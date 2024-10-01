import typer
from scratchpad.server import dataclass_to_cli, ServerArgs, launch_server
from scratchpad.server.args import global_args
from .handlers import ChatHandler

app = typer.Typer()


@app.command()
@dataclass_to_cli
def serve(
    model: str,
    args: ServerArgs,
):
    """Spin up the server"""
    typer.echo(f"Serving model: {model}, args: {args}")
    global_args = args
    launch_server(model, args)


@app.command()
def version():
    """Print the version"""
    typer.echo("0.1.0")


@app.command()
def chat(
    model: str,
    backend: str = "http://localhost:8080",
):
    chat_handler = ChatHandler(server_addr=backend, model_name=model)
    chat_handler.chat()


if __name__ == "__main__":
    app()
