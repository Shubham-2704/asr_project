"""
main.py - Master Pipeline Runner
================================
Connects all modules and runs the complete Hindi ASR pipeline.

Execution flow:
  1. preprocess   → Download data + prepare dataset
  2. train        → Fine-tune Whisper-small on Hindi
  3. evaluate     → Compute WER on FLEURS Hindi test set
  4. error_analysis → Sample errors + build taxonomy + propose fixes
  5. cleanup      → Number normalization pipeline (Q2a)
  6. english_detect → English word detection + tagging (Q2b)
  7. spelling     → Hindi spelling classification (Q3)
  8. lattice      → Lattice-based WER evaluation (Q4)

Usage:
    python main.py                          # Run ALL steps
    python main.py --step preprocess        # Run only preprocessing
    python main.py --step train             # Run only training
    python main.py --step evaluate          # Run only evaluation
    python main.py --step lattice           # Run only lattice (Q4)
    python main.py --step spelling          # Run only spelling (Q3)
    python main.py --steps cleanup,english_detect  # Run multiple steps
"""

import os
import sys
import time
import argparse
import importlib
import traceback


# ===== Step Definitions =====
# Each step maps to a module in src/ and has a description
STEPS = {
    "preprocess": {
        "module": "src.preprocess",
        "description": "Download audio + transcriptions, build HuggingFace dataset",
        "question": "Q1a",
    },
    "train": {
        "module": "src.train",
        "description": "Fine-tune Whisper-small on Hindi ASR data (CPU optimized)",
        "question": "Q1b",
    },
    "evaluate": {
        "module": "src.evaluate",
        "description": "Compute WER on FLEURS Hindi test set",
        "question": "Q1b-c",
    },
    "error_analysis": {
        "module": "src.error_analysis",
        "description": "Sample errors, build taxonomy, propose & implement fixes",
        "question": "Q1d-g",
    },
    "cleanup": {
        "module": "src.cleanup",
        "description": "Number normalization for ASR output",
        "question": "Q2a",
    },
    "english_detect": {
        "module": "src.english_detect",
        "description": "Detect and tag English words in Hindi transcripts",
        "question": "Q2b",
    },
    "spelling": {
        "module": "src.spelling",
        "description": "Classify 1,75,000 Hindi words as correct/incorrect spelling",
        "question": "Q3",
    },
    "lattice": {
        "module": "src.lattice",
        "description": "Lattice-based WER evaluation with multi-model consensus",
        "question": "Q4",
    },
}

# Default execution order
DEFAULT_ORDER = [
    "preprocess", "train", "evaluate", "error_analysis",
    "cleanup", "english_detect", "spelling", "lattice",
]


def print_banner():
    """Print the project banner."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║   Hindi ASR Pipeline — Josh Talks Assignment             ║")
    print("║   Speech & Audio Research Intern                         ║")
    print("╠" + "═" * 58 + "╣")
    print("║   Q1: Whisper Fine-tuning + WER + Error Analysis         ║")
    print("║   Q2: ASR Cleanup (Numbers + English Detection)          ║")
    print("║   Q3: Hindi Spelling Classification                      ║")
    print("║   Q4: Lattice-Based WER Evaluation                       ║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_step_list():
    """Print available steps."""
    print("Available steps:")
    print(f"{'Step':<20} {'Question':<10} {'Description'}")
    print("-" * 70)
    for name in DEFAULT_ORDER:
        info = STEPS[name]
        print(f"  {name:<18} {info['question']:<10} {info['description']}")
    print()


def run_step(step_name: str) -> bool:
    """
    Run a single pipeline step.

    Args:
        step_name: Name of the step to run

    Returns:
        True if step completed successfully, False otherwise
    """
    if step_name not in STEPS:
        print(f"[✗] Unknown step: '{step_name}'")
        print_step_list()
        return False

    step_info = STEPS[step_name]
    print()
    print("╔" + "═" * 58 + "╗")
    print(f"║  Running: {step_name:<47} ║")
    print(f"║  {step_info['question']}: {step_info['description'][:44]:<44} ║")
    print("╚" + "═" * 58 + "╝")
    print()

    start_time = time.time()

    try:
        # Import and run the module's main() function
        module = importlib.import_module(step_info["module"])

        # Reset sys.argv so argparse in the module doesn't fail
        original_argv = sys.argv
        sys.argv = [step_info["module"]]

        module.main()

        sys.argv = original_argv

        elapsed = time.time() - start_time
        print(f"\n[✓] Step '{step_name}' completed in {elapsed:.1f}s")
        return True

    except SystemExit as e:
        elapsed = time.time() - start_time
        if e.code == 0:
            print(f"\n[✓] Step '{step_name}' completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n[✗] Step '{step_name}' exited with code {e.code}")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[✗] Step '{step_name}' failed after {elapsed:.1f}s")
        print(f"    Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Hindi ASR Pipeline — Complete Assignment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run all steps
  python main.py --step preprocess        # Run one step
  python main.py --steps train,evaluate   # Run multiple steps
  python main.py --list                   # List available steps
  python main.py --step lattice           # Run Q4 only (no GPU needed)
  python main.py --step spelling          # Run Q3 only (no GPU needed)
        """,
    )
    parser.add_argument("--step", type=str, default=None,
                        help="Run a single step (e.g., 'preprocess', 'train')")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated list of steps to run")
    parser.add_argument("--list", action="store_true",
                        help="List all available steps")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue running next steps even if one fails")
    args = parser.parse_args()

    print_banner()

    # List steps
    if args.list:
        print_step_list()
        return

    # Determine which steps to run
    if args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = [s.strip() for s in args.steps.split(",")]
    else:
        steps_to_run = DEFAULT_ORDER

    # Validate steps
    for step in steps_to_run:
        if step not in STEPS:
            print(f"[✗] Unknown step: '{step}'")
            print_step_list()
            sys.exit(1)

    # Run steps
    print(f"[→] Running {len(steps_to_run)} step(s): {', '.join(steps_to_run)}")
    print()

    results = {}
    total_start = time.time()

    for i, step_name in enumerate(steps_to_run):
        print(f"\n{'─' * 60}")
        print(f"  Step {i+1}/{len(steps_to_run)}: {step_name}")
        print(f"{'─' * 60}")

        success = run_step(step_name)
        results[step_name] = "✓ Passed" if success else "✗ Failed"

        if not success and not args.continue_on_error:
            print(f"\n[!] Pipeline stopped at step '{step_name}'.")
            print(f"[i] Use --continue-on-error to skip failed steps.")
            break

    # Print summary
    total_elapsed = time.time() - total_start

    print(f"\n{'═' * 60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'═' * 60}")
    for step_name, status in results.items():
        q = STEPS[step_name]["question"]
        print(f"  {step_name:<20} ({q:<5}) {status}")
    print(f"{'─' * 60}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'═' * 60}")

    # Check for any failures
    failures = [s for s, r in results.items() if "Failed" in r]
    if failures:
        print(f"\n[!] {len(failures)} step(s) failed: {', '.join(failures)}")
        print(f"[i] Check the output above for error details.")
        sys.exit(1)
    else:
        print(f"\n[✓] All steps completed successfully!")
        print(f"    Results are in: outputs/")


if __name__ == "__main__":
    main()
