"""
Command Line Interface (CLI) for sqst.
"""

import sys
from typing import Optional

import click
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sqst import __version__
from sqst.pipeline import load
from sqst.registry import (
    get_model_info,
    is_cached_locally,
    list_models,
    resolve_model_id,
)

console = Console()

BACKEND_CHOICES = click.Choice(["pytorch", "onnx", "onnx-quantized"])


def _print_banner(title: str, subtitle: Optional[str] = None) -> None:
    text = Text()
    text.append(title, style="bold")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim")
    console.print(Panel(text, border_style="dim", padding=(0, 2)))


def _load_or_exit(model_name: str, device: str, backend: str, trust_remote_code: bool, label: str):
    """Loads a model with a spinner and exits with a readable message on failure."""
    try:
        with console.status(f"{label} '{escape(model_name)}'...", spinner="dots"):
            return load(
                model_name,
                device=device,
                backend=backend,
                trust_remote_code=True if trust_remote_code else None,
            )
    except Exception as e:  # noqa: BLE001 - show any load failure to the user
        console.print(f"[bold red]Failed to load model:[/bold red] {escape(str(e))}")
        sys.exit(1)


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version=__version__, prog_name="sqst")
def main(ctx: click.Context) -> None:
    """sqst: a convenient tool and library for running Squeal Studio models."""
    if ctx.invoked_subcommand is None:
        console.print("[bold]SQST[/bold] — inference library for Squeal Studio models.")
        console.print("Use [bold]sqst --help[/bold] to see the list of commands.\n")
        ctx.invoke(list_cmd)


@main.command(name="list")
def list_cmd() -> None:
    """List the available Squeal Studio models."""
    table = Table(title="Squeal Studio models", border_style="dim", header_style="bold", title_style="bold")
    table.add_column("Alias", style="bold", no_wrap=True)
    table.add_column("Repository", no_wrap=False)
    table.add_column("Architecture")
    table.add_column("Type")
    table.add_column("Parameters", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Cached", justify="center")

    for m in list_models():
        cached = is_cached_locally(m.repo_id)
        cache_badge = "yes" if cached else "[dim]no[/dim]"
        primary_alias = m.aliases[0]
        cat_badge = "instruct" if m.is_instruct else "[dim]base[/dim]"
        arch_display = {
            "qwen2": "Qwen2",
            "drakon": "Drakon (Hybrid)",
            "lstm_lm": "LSTM",
        }.get(m.arch, m.arch)

        table.add_row(
            primary_alias,
            m.repo_id,
            arch_display,
            cat_badge,
            m.parameters,
            str(m.context_length),
            cache_badge,
        )

    console.print(table)
    console.print("[dim]Tip: use a short alias to start a model (e.g. sqst chat 8m or sqst run drakon)[/dim]\n")


@main.command(name="info")
@click.argument("model_name")
def info_cmd(model_name: str) -> None:
    """Show detailed information about a model."""
    try:
        repo_id = resolve_model_id(model_name)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
        sys.exit(1)
    info = get_model_info(repo_id)

    if not info:
        console.print(f"[bold red]Error:[/bold red] model '{escape(model_name)}' was not found in the catalog.")
        sys.exit(1)

    cached = is_cached_locally(info.repo_id)
    panel_content = (
        f"[dim]Name:[/dim] {escape(info.name)}\n"
        f"[dim]Repository:[/dim] {escape(info.repo_id)}\n"
        f"[dim]Aliases:[/dim] {escape(', '.join(info.aliases))}\n"
        f"[dim]Architecture:[/dim] {info.arch}\n"
        f"[dim]Category:[/dim] {info.category}\n"
        f"[dim]Parameters:[/dim] {info.parameters}\n"
        f"[dim]Context (tokens):[/dim] {info.context_length}\n"
        f"[dim]ONNX support:[/dim] {'Yes' if info.has_onnx else 'No'}\n"
        f"[dim]Cached locally:[/dim] {'Yes' if cached else 'No'}\n\n"
        f"[bold]Description[/bold]\n{escape(info.description)}\n\n"
        f"[bold]Default generation parameters[/bold]\n{escape(str(info.default_gen_args))}"
    )

    console.print(Panel(panel_content, title=f"{escape(info.name)}", border_style="dim"))


@main.command(name="download")
@click.argument("model_name")
def download_cmd(model_name: str) -> None:
    """Download model weights into the local Hugging Face cache."""
    try:
        repo_id = resolve_model_id(model_name)
        console.print(f"Downloading weights for '{escape(repo_id)}'...")
        snapshot_download(repo_id=repo_id)
        console.print(f"Weights for '{escape(repo_id)}' were downloaded and cached.")
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Download failed:[/bold red] {escape(str(e))}")
        sys.exit(1)


@main.command(name="run")
@click.argument("model_name")
@click.argument("prompt", required=False)
@click.option("--max-tokens", "-m", default=None, type=int, help="Maximum number of new tokens.")
@click.option("--temperature", "-t", default=None, type=float, help="Sampling temperature.")
@click.option("--top-p", "-p", default=None, type=float, help="Top-p (nucleus) sampling.")
@click.option("--top-k", "-k", default=None, type=int, help="Top-k sampling.")
@click.option("--repetition-penalty", "-r", default=None, type=float, help="Repetition penalty.")
@click.option("--device", "-d", default="auto", help="Device: auto, cpu, cuda, mps.")
@click.option("--backend", "-b", default="pytorch", type=BACKEND_CHOICES, help="Inference backend.")
@click.option("--trust-remote-code", is_flag=True, help="Allow executing code from a non-catalog model repo.")
@click.option("--stream/--no-stream", default=True, help="Enable streaming output.")
@click.option("--raw", is_flag=True, help="Disable chat-template formatting.")
def run_cmd(
    model_name: str,
    prompt: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    device: str,
    backend: str,
    trust_remote_code: bool,
    stream: bool,
    raw: bool,
) -> None:
    """Generate a response to a prompt."""
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            prompt = click.prompt("Enter a prompt")

    if not prompt or not prompt.strip():
        console.print("[bold red]Error:[/bold red] the prompt is empty.")
        sys.exit(1)

    model = _load_or_exit(model_name, device, backend, trust_remote_code, "Loading model")
    console.print(f"[dim]Model {escape(model.model_id)} ({model.device})[/dim]\n")

    gen_args = dict(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        raw=raw,
    )

    try:
        if stream:
            for chunk in model.stream(prompt, **gen_args):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            # click.echo prints the text verbatim (console.print would parse "[...]" as markup).
            click.echo(model.generate(prompt, **gen_args))
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[bold red]Generation error:[/bold red] {escape(str(e))}")
        sys.exit(1)


@main.command(name="chat")
@click.argument("model_name", default="8m")
@click.option("--device", "-d", default="auto", help="Device: auto, cpu, cuda, mps.")
@click.option("--backend", "-b", default="pytorch", type=BACKEND_CHOICES, help="Inference backend.")
@click.option("--trust-remote-code", is_flag=True, help="Allow executing code from a non-catalog model repo.")
@click.option("--system", "-s", default=None, help="System prompt (instruction).")
@click.option("--temperature", "-t", default=None, type=float, help="Sampling temperature.")
@click.option("--max-tokens", "-m", default=None, type=int, help="Maximum number of new tokens per reply.")
def chat_cmd(
    model_name: str,
    device: str,
    backend: str,
    trust_remote_code: bool,
    system: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> None:
    """Start an interactive chat with a model in the terminal."""
    model = _load_or_exit(model_name, device, backend, trust_remote_code, "Initializing model")

    session = model.chat(system_prompt=system)

    _print_banner(
        f"Chat with {model.model_id}",
        subtitle=f"Device: {model.device} | Type /help for commands, /exit to quit.",
    )

    while True:
        try:
            user_input = console.input("\n[bold]You >[/bold] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("//"):
            user_input = user_input[1:]  # "//path" sends a literal message starting with "/"
        elif user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd in ("/clear", "/reset"):
                session.reset()
                console.print("Conversation history cleared.")
                continue
            elif cmd in ("/history", "/h"):
                console.print(f"Messages in history: {len(session.history)}")
                for msg in session.history:
                    console.print(f"[bold]{msg.role}[/bold]: {escape(msg.content)}")
                continue
            elif cmd in ("/undo", "/u"):
                popped = session.undo()
                if popped:
                    console.print(f"Removed last turn: '{escape(popped.content)}'")
                else:
                    console.print("[dim]History is empty.[/dim]")
                continue
            elif cmd in ("/help", "/?"):
                console.print("[bold]Available commands:[/bold]")
                console.print("  /clear   — Clear the conversation history")
                console.print("  /history — Show the message history")
                console.print("  /undo    — Remove the last turn")
                console.print("  /exit    — Quit the chat")
                console.print("  //text   — Send a message that starts with a slash")
                continue
            else:
                console.print(f"[dim]Unknown command {escape(user_input)}. Type /help[/dim]")
                continue

        label = model.info.name if model.info else "Assistant"
        console.print(f"[bold]{escape(label)} >[/bold] ", end="")
        reply_stream = session.stream(
            user_input,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        try:
            for chunk in reply_stream:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
        except KeyboardInterrupt:
            console.print("\n[dim]Generation interrupted.[/dim]")
        except Exception as e:  # noqa: BLE001
            console.print(f"\n[bold red]Generation error:[/bold red] {escape(str(e))}")
        finally:
            reply_stream.close()  # keeps the history consistent after an interrupt


if __name__ == "__main__":
    main()
