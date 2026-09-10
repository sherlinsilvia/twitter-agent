import json
import time
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.dataset_loader import DatasetLoader
from src.baselines import TrivialBaseline, SimpleBaseline
from src.agent import SupportAgent
from src.evaluator import Evaluator

def main():
    console = Console()
    console.print("\n[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold cyan]   Hiver AI Customer Support Agent - Benchmark Pipeline    [/bold cyan]")
    console.print("[bold cyan]===========================================================\n[/bold cyan]")

    start_time = time.time()
    
    # 1. Load Data
    console.print("[bold yellow][1/4] Loading datasets...[/bold yellow]")
    loader = DatasetLoader(data_dir="data")
    golden_eval_set = loader.load_golden_eval_set()
    raw_df = loader.load_raw_samples()
    knowledge_base = loader.load_historical_knowledge()
    console.print(f" -> Loaded {len(golden_eval_set)} Golden Evaluation Set examples.")
    console.print(f" -> Loaded {len(raw_df)} historical training tweets.")
    console.print(f" -> Loaded {len(knowledge_base)} grounded KB resolution articles.\n")

    # 2. Instantiate Models
    console.print("[bold yellow][2/4] Initializing models...[/bold yellow]")
    trivial_model = TrivialBaseline()
    simple_model = SimpleBaseline(raw_df, knowledge_base)
    main_agent = SupportAgent(data_dir="data")
    console.print(" -> Trivial Baseline initialized.")
    console.print(" -> Simple Baseline initialized.")
    console.print(" -> Main AI Support Agent (Classifier + RAG + Guardrail Escalation) initialized.\n")

    # 3. Evaluate All Models
    console.print("[bold yellow][3/4] Running Golden Evaluation Set benchmarks...[/bold yellow]")
    evaluator = Evaluator()

    trivial_res = evaluator.evaluate_pipeline(trivial_model, golden_eval_set)
    simple_res = evaluator.evaluate_pipeline(simple_model, golden_eval_set)
    main_res = evaluator.evaluate_pipeline(main_agent, golden_eval_set)

    # 4. Display Comparison Table
    console.print("\n[bold yellow][4/4] Benchmark Results Summary:[/bold yellow]\n")

    table = Table(title="Hiver AI Support Agent - Headline Evaluation Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline 1 (Trivial)", style="magenta")
    table.add_column("Baseline 2 (Simple)", style="blue")
    table.add_column("Main AI Agent", style="bold green")

    table.add_row(
        "Intent Accuracy",
        f"{trivial_res['intent_classification']['accuracy']:.2%}",
        f"{simple_res['intent_classification']['accuracy']:.2%}",
        f"[bold]{main_res['intent_classification']['accuracy']:.2%}[/bold]"
    )
    table.add_row(
        "Intent Macro F1",
        f"{trivial_res['intent_classification']['macro_f1']:.4f}",
        f"{simple_res['intent_classification']['macro_f1']:.4f}",
        f"[bold]{main_res['intent_classification']['macro_f1']:.4f}[/bold]"
    )
    table.add_row(
        "Escalation Precision",
        f"{trivial_res['escalation_routing']['precision']:.2%}",
        f"{simple_res['escalation_routing']['precision']:.2%}",
        f"[bold]{main_res['escalation_routing']['precision']:.2%}[/bold]"
    )
    table.add_row(
        "Escalation Recall",
        f"{trivial_res['escalation_routing']['recall']:.2%}",
        f"{simple_res['escalation_routing']['recall']:.2%}",
        f"[bold]{main_res['escalation_routing']['recall']:.2%}[/bold]"
    )
    table.add_row(
        "Reply Overlap (Jaccard)",
        f"{trivial_res['reply_quality']['avg_jaccard_similarity']:.4f}",
        f"{simple_res['reply_quality']['avg_jaccard_similarity']:.4f}",
        f"[bold]{main_res['reply_quality']['avg_jaccard_similarity']:.4f}[/bold]"
    )
    table.add_row(
        "LLM-as-Judge Score (1-5)",
        f"{trivial_res['reply_quality']['llm_judge_avg_score']}/5.0",
        f"{simple_res['reply_quality']['llm_judge_avg_score']}/5.0",
        f"[bold]{main_res['reply_quality']['llm_judge_avg_score']}/5.0[/bold]"
    )
    table.add_row(
        "Human Judge Agreement",
        f"{trivial_res['reply_quality']['human_judge_agreement_correlation']:.4f}",
        f"{simple_res['reply_quality']['human_judge_agreement_correlation']:.4f}",
        f"[bold]{main_res['reply_quality']['human_judge_agreement_correlation']:.4f}[/bold]"
    )

    console.print(table)

    elapsed = time.time() - start_time
    console.print(f"\n[bold green][OK] Benchmark complete in {elapsed:.2f} seconds![/bold green]\n")

    # Save output json
    results_summary = {
        "execution_time_seconds": round(elapsed, 2),
        "trivial_baseline": trivial_res,
        "simple_baseline": simple_res,
        "main_agent": main_res
    }
    
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    console.print("Results saved to [bold]results.json[/bold].\n")

if __name__ == "__main__":
    main()
