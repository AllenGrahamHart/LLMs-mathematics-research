#!/usr/bin/env python3
"""
Batch experiment runner for grouped papers - each experiment can include multiple papers as context.
Handles errors, tracks progress, and can be resumed.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def load_progress(progress_file):
    """Load progress from file if it exists."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress_file, progress):
    """Save progress to file."""
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def load_config(config_file):
    """Load experiment configuration from JSON file."""
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"Error: Config file '{config_file}' not found")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

    return config


def validate_experiments(experiments):
    """Validate experiment configuration."""
    if not experiments:
        print("Error: No experiments defined in config file")
        sys.exit(1)

    papers_dir = Path("problems/papers")
    experiment_names = set()
    errors = []

    for i, exp in enumerate(experiments, 1):
        exp_id = f"Experiment {i}"

        # Check required fields
        if 'name' not in exp:
            errors.append(f"{exp_id}: Missing required field 'name'")
            continue

        exp_name = exp['name']
        exp_id = f"Experiment '{exp_name}'"

        # Check for duplicate names
        if exp_name in experiment_names:
            errors.append(f"{exp_id}: Duplicate experiment name")
        experiment_names.add(exp_name)

        # Check papers field
        if 'papers' not in exp:
            errors.append(f"{exp_id}: Missing required field 'papers'")
            continue

        papers = exp['papers']

        # Check papers is a list
        if not isinstance(papers, list):
            errors.append(f"{exp_id}: 'papers' must be a list")
            continue

        # Check papers list is not empty
        if not papers:
            errors.append(f"{exp_id}: 'papers' list cannot be empty")
            continue

        # Check that paper files exist
        for paper in papers:
            paper_file = papers_dir / f"{paper}.txt"
            if not paper_file.exists():
                errors.append(f"{exp_id}: Paper file not found: {paper}.txt")

    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def run_experiment(experiment_name, papers, max_iterations, dry_run=False):
    """Run a single experiment with one or more papers."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Papers: {', '.join(papers)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if dry_run:
        paper_args = ' '.join(papers)
        print(f"[DRY RUN] Would run: python run_experiment.py --papers {paper_args} --max-iterations {max_iterations}")
        return True

    cmd = [
        "python", "run_experiment.py",
        "--papers", *papers,
        "--max-iterations", str(max_iterations)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Successfully completed: {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {experiment_name}")
        print(f"Error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user during: {experiment_name}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments with grouped papers')
    parser.add_argument('--config', type=str, required=True, help='JSON config file (in batch_configs/) with experiment definitions')
    parser.add_argument('--progress-file', type=str, default=None, help='File to track progress (default: batch_progress/<config_name>_progress.json)')
    parser.add_argument('--skip-completed', action='store_true', help='Skip experiments that have already been completed')
    parser.add_argument('--dry-run', action='store_true', help='Simulate running without actually executing')
    parser.add_argument('--start-from', type=str, help='Start from a specific experiment name')
    args = parser.parse_args()

    # Resolve config path (check batch_configs/ if not found as-is)
    config_path = args.config
    if not Path(config_path).exists():
        # Try batch_configs/ directory
        config_path = f"batch_configs/{args.config}"
        if not Path(config_path).exists():
            print(f"Error: Config file not found: {args.config}")
            print(f"  Tried: {args.config}")
            print(f"  Tried: batch_configs/{args.config}")
            sys.exit(1)

    # Load configuration
    config = load_config(config_path)
    experiments = config.get('experiments', [])

    # Validate experiment configuration
    validate_experiments(experiments)

    # Setup progress tracking
    if args.progress_file:
        progress_file = Path(args.progress_file)
    else:
        # Auto-generate progress filename from config name
        config_name = Path(config_path).stem
        progress_file = Path(f"batch_progress/{config_name}_progress.json")
        # Ensure batch_progress directory exists
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    progress = load_progress(progress_file)

    # Filter based on start_from
    if args.start_from:
        experiment_names = [exp['name'] for exp in experiments]
        try:
            start_idx = experiment_names.index(args.start_from)
            experiments = experiments[start_idx:]
            print(f"Starting from experiment: {args.start_from}")
        except ValueError:
            print(f"Error: Experiment '{args.start_from}' not found in config")
            sys.exit(1)

    # Filter out already completed if requested
    if args.skip_completed:
        original_count = len(experiments)
        experiments = [exp for exp in experiments if exp['name'] not in progress["completed"]]
        skipped = original_count - len(experiments)
        if skipped > 0:
            print(f"Skipping {skipped} already completed experiments")

    # Check if any experiments remain after filtering
    if not experiments:
        print("No experiments to run (all completed or filtered out)")
        sys.exit(0)

    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH EXPERIMENT RUNNER (GROUPED)")
    print(f"{'='*60}")
    print(f"Total experiments to run: {len(experiments)}")
    print(f"Progress file: {progress_file}")
    print(f"Dry run: {args.dry_run}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show experiment list
    print(f"\nExperiments:")
    for i, exp in enumerate(experiments, 1):
        papers_str = ', '.join(exp['papers'])
        print(f"  {i}. {exp['name']}: [{papers_str}] ({exp.get('max_iterations', 10)} iterations)")

    # Estimate time (rough average per experiment regardless of paper count)
    est_hours = len(experiments) * 1.75
    print(f"\nEstimated time: {est_hours:.1f} hours ({est_hours/24:.1f} days)")
    print(f"{'='*60}\n")

    if not args.dry_run:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Run experiments
    start_time = datetime.now()
    completed_count = 0
    failed_count = 0

    try:
        for i, exp in enumerate(experiments, 1):
            exp_name = exp['name']
            papers = exp['papers']
            max_iterations = exp.get('max_iterations', 10)

            print(f"\n[{i}/{len(experiments)}] Processing experiment: {exp_name}")

            success = run_experiment(exp_name, papers, max_iterations, args.dry_run)

            if success:
                progress["completed"].append(exp_name)
                completed_count += 1
            else:
                progress["failed"].append(exp_name)
                failed_count += 1

            # Save progress after each experiment
            save_progress(progress_file, progress)

            # Print summary
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            remaining = len(experiments) - i
            avg_time = elapsed / i
            est_remaining = remaining * avg_time

            print(f"\nProgress: {i}/{len(experiments)} experiments")
            print(f"Completed: {completed_count}, Failed: {failed_count}")
            print(f"Elapsed: {elapsed:.1f}h, Remaining: {est_remaining:.1f}h")

    except KeyboardInterrupt:
        print("\n\n⚠️  Batch run interrupted by user")
        print("Progress has been saved. You can resume with:")
        print(f"  python run_batch_experiments_grouped.py --config {Path(config_path).name} --skip-completed")
        sys.exit(1)

    # Final summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 3600

    print(f"\n{'='*60}")
    print(f"BATCH RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.1f} hours ({total_time/24:.1f} days)")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if progress["failed"]:
        print("Failed experiments:")
        for exp_name in progress["failed"]:
            print(f"  - {exp_name}")
        print()

    print(f"Progress saved to: {progress_file}")


if __name__ == "__main__":
    main()
