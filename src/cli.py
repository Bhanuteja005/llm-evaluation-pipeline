"""Command-line interface for LLM evaluation pipeline."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.evaluators.factual import FactualEvaluator
from src.evaluators.latency_cost import LatencyCostEvaluator
from src.evaluators.relevance import RelevanceEvaluator
from src.ingest import load_inputs
from src.llm_client import get_llm_client
from src.persistence import Database
from src.prompt_builder import build_prompt
from src.scoring import Scorer
from src.utils import get_logger

logger = get_logger(__name__)
console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """LLM Evaluation Pipeline - Evaluate LLM responses for relevance, accuracy, and performance."""
    pass


@cli.command()
@click.option(
    "--conversation",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to conversation JSON file",
)
@click.option(
    "--context",
    "-x",
    required=True,
    type=click.Path(exists=True),
    help="Path to context vectors JSON file",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(),
    help="Output JSON file path (optional)",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["mock", "gemini"]),
    default=None,
    help="LLM provider (defaults to env setting)",
)
@click.option(
    "--save-db/--no-save-db",
    default=True,
    help="Save results to database",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
def evaluate(
    conversation: str,
    context: str,
    out: Optional[str],
    provider: Optional[str],
    save_db: bool,
    verbose: bool,
) -> None:
    """
    Evaluate LLM response given conversation and context.

    Example:
        llm-eval evaluate -c conversation.json -x context.json -o results.json
    """
    try:
        console.print("\n[bold cyan]LLM Evaluation Pipeline[/bold cyan]\n")

        # Load inputs
        with console.status("[bold green]Loading input files..."):
            conversation_data, context_data = load_inputs(conversation, context)
            console.print("✓ Input files loaded successfully")

        # Get latest user message
        latest_message = conversation_data.get_latest_user_message()
        if not latest_message:
            console.print("[bold red]Error: No user message found in conversation[/bold red]")
            sys.exit(1)

        console.print(f"Chat ID: {conversation_data.chat_id}")
        console.print(f"Turn: {latest_message.turn}")
        console.print(f"User message: {latest_message.message[:100]}...")

        # Get context vectors
        context_vectors = context_data.get_vector_data()
        console.print(f"Context vectors: {len(context_vectors)}")

        # Build prompt
        with console.status("[bold green]Building prompt..."):
            prompt = build_prompt(conversation_data, context_vectors[:settings.top_k_contexts])
            if verbose:
                console.print(f"\n[dim]Prompt length: {len(prompt)} characters[/dim]")

        # Get LLM response
        with console.status("[bold green]Generating LLM response..."):
            llm_client = get_llm_client(provider)
            llm_response = llm_client.generate(prompt)
            console.print(f"✓ LLM response generated ({llm_response.latency_ms:.0f}ms)")

        if verbose:
            console.print(f"\n[dim]Response: {llm_response.text[:200]}...[/dim]\n")

        # Initialize evaluators
        with console.status("[bold green]Initializing evaluators..."):
            relevance_evaluator = RelevanceEvaluator()
            relevance_evaluator.build_index(context_vectors)

            factual_evaluator = FactualEvaluator()
            factual_evaluator.set_context(context_vectors)

            latency_cost_evaluator = LatencyCostEvaluator()

            scorer = Scorer()
            console.print("✓ Evaluators initialized")

        # Run evaluations
        console.print("\n[bold yellow]Running Evaluations...[/bold yellow]\n")

        with console.status("[bold green]Evaluating relevance..."):
            relevance_result = relevance_evaluator.evaluate(
                llm_response.text, latest_message.message
            )
            console.print("✓ Relevance evaluation complete")

        with console.status("[bold green]Evaluating factual accuracy..."):
            factual_result = factual_evaluator.evaluate(llm_response.text)
            console.print("✓ Factual evaluation complete")

        with console.status("[bold green]Evaluating latency and cost..."):
            latency_cost_result = latency_cost_evaluator.evaluate(
                llm_response.latency_ms,
                llm_response.token_count,
                llm_response.input_tokens,
                llm_response.output_tokens,
                llm_response.estimated_cost_usd(),
            )
            console.print("✓ Latency/cost evaluation complete")

        # Create report
        report = scorer.create_report(
            chat_id=conversation_data.chat_id,
            turn=latest_message.turn,
            user_message=latest_message.message,
            model_response=llm_response.text,
            provider=llm_response.provider,
            relevance=relevance_result,
            factual=factual_result,
            latency_cost=latency_cost_result,
        )

        # Display results
        _display_results(report)

        # Save to database
        if save_db:
            with console.status("[bold green]Saving to database..."):
                db = Database()
                record_id = db.save_evaluation(report)
                console.print(f"✓ Saved to database (ID: {record_id})")

        # Save to JSON file
        if out:
            output_path = Path(out)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            console.print(f"✓ Results saved to: {output_path}")

        console.print("\n[bold green]Evaluation complete![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--chat-id",
    "-c",
    type=int,
    help="Chat ID to query",
)
def stats(chat_id: Optional[int]) -> None:
    """Display evaluation statistics."""
    try:
        db = Database()

        if chat_id:
            console.print(f"\n[bold cyan]Statistics for Chat {chat_id}[/bold cyan]\n")
            records = db.get_evaluations_by_chat(chat_id)

            if not records:
                console.print(f"No evaluations found for chat {chat_id}")
                return

            table = Table(show_header=True)
            table.add_column("Turn", style="cyan")
            table.add_column("Quality", style="green")
            table.add_column("Hallucination", style="yellow")
            table.add_column("Latency (ms)", style="blue")
            table.add_column("Passed", style="magenta")

            for record in records:
                table.add_row(
                    str(record.turn),
                    f"{record.overall_quality_score:.3f}",
                    f"{record.hallucination_rate:.3f}",
                    f"{record.latency_ms:.0f}",
                    "✓" if record.passed_thresholds else "✗",
                )

            console.print(table)

        else:
            console.print("\n[bold cyan]Overall Statistics[/bold cyan]\n")
            stats_data = db.get_statistics()

            table = Table(show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Evaluations", str(stats_data["total_evaluations"]))
            table.add_row("Avg Quality Score", f"{stats_data['avg_quality_score']:.3f}")
            table.add_row(
                "Avg Hallucination Rate", f"{stats_data['avg_hallucination_rate']:.3f}"
            )
            table.add_row("Avg Latency (ms)", f"{stats_data['avg_latency_ms']:.0f}")
            table.add_row("Pass Rate", f"{stats_data['pass_rate']:.1f}%")

            console.print(table)

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        sys.exit(1)


def _display_results(report) -> None:
    """Display evaluation results in a formatted way."""
    # Overall summary
    summary_text = "\n".join(report.summary)
    console.print(
        Panel(
            summary_text,
            title="[bold]Evaluation Summary[/bold]",
            border_style="cyan",
        )
    )

    # Metrics table
    table = Table(show_header=True, title="Detailed Metrics")
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Metric", style="yellow", width=25)
    table.add_column("Value", style="green", width=15)

    # Relevance metrics
    table.add_row(
        "Relevance",
        "Relevance Score",
        f"{report.relevance.relevance_score:.3f}",
    )
    table.add_row(
        "",
        "Completeness Score",
        f"{report.relevance.completeness_score:.3f}",
    )

    # Factual metrics
    table.add_row(
        "Factual Accuracy",
        "Hallucination Rate",
        f"{report.factual.hallucination_rate:.3f}",
    )
    table.add_row(
        "",
        "Verified Claims",
        f"{report.factual.verified_claims}/{report.factual.total_claims}",
    )

    # Performance metrics
    table.add_row(
        "Performance",
        "Latency (ms)",
        f"{report.latency_cost.latency_ms:.0f}",
    )
    table.add_row(
        "",
        "Estimated Cost (USD)",
        f"${report.latency_cost.estimated_cost_usd:.6f}",
    )
    table.add_row(
        "",
        "Token Count",
        str(report.latency_cost.token_count),
    )

    console.print("\n", table, "\n")


def main() -> None:
    """Main entrypoint."""
    cli()


if __name__ == "__main__":
    main()
