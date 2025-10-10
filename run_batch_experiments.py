#!/usr/bin/env python3
"""
Batch experiment runner - processes multiple papers sequentially.
Handles errors, tracks progress, and can be resumed.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import time


def load_progress(progress_file):
    """Load progress from file if it exists."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "skipped": []}


def save_progress(progress_file, progress):
    """Save progress to file."""
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def get_paper_list():
    """Get list of all paper files."""
    papers_dir = Path("problems/papers")
    if not papers_dir.exists():
        print(f"Error: {papers_dir} not found")
        sys.exit(1)

    # Get all .txt files and strip the .txt extension
    papers = sorted([p.stem for p in papers_dir.glob("*.txt")])

    if not papers:
        print(f"Error: No paper files found in {papers_dir}")
        sys.exit(1)

    return papers


def run_experiment(paper_id, max_iterations, dry_run=False):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {paper_id}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if dry_run:
        print(f"[DRY RUN] Would run: python run_experiment.py --papers {paper_id} --max-iterations {max_iterations}")
        time.sleep(2)  # Simulate some work
        return True

    cmd = [
        "python", "run_experiment.py",
        "--papers", paper_id,
        "--max-iterations", str(max_iterations)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Successfully completed: {paper_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {paper_id}")
        print(f"Error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user during: {paper_id}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Run experiments for multiple papers sequentially')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum iterations per experiment (default: 10)')
    parser.add_argument('--progress-file', type=str, default='batch_progress/single.json', help='File to track progress (default: batch_progress/single.json)')
    parser.add_argument('--skip-completed', action='store_true', help='Skip papers that have already been completed')
    parser.add_argument('--papers', nargs='+', help='Specific papers to run (default: all papers in problems/papers/)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate running without actually executing')
    parser.add_argument('--start-from', type=str, help='Start from a specific paper (useful for resuming)')
    args = parser.parse_args()

    # Setup
    progress_file = Path(args.progress_file)
    # Ensure batch_progress directory exists
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress = load_progress(progress_file)

    # Get paper list
    if args.papers:
        papers = args.papers
    else:
        papers = get_paper_list()

    # Filter based on start_from
    if args.start_from:
        try:
            start_idx = papers.index(args.start_from)
            papers = papers[start_idx:]
            print(f"Starting from paper: {args.start_from}")
        except ValueError:
            print(f"Error: Paper '{args.start_from}' not found in list")
            sys.exit(1)

    # Filter out already completed if requested
    if args.skip_completed:
        original_count = len(papers)
        papers = [p for p in papers if p not in progress["completed"]]
        skipped = original_count - len(papers)
        if skipped > 0:
            print(f"Skipping {skipped} already completed papers")

    # Check if any papers remain after filtering
    if not papers:
        print("No papers to run (all completed or filtered out)")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"BATCH EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"Total papers to process: {len(papers)}")
    print(f"Max iterations per paper: {args.max_iterations}")
    print(f"Progress file: {progress_file}")
    print(f"Dry run: {args.dry_run}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Estimate time
    est_hours = len(papers) * 1.75  # Average 1.75 hours per paper
    print(f"Estimated time: {est_hours:.1f} hours ({est_hours/24:.1f} days)")
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
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing: {paper}")

            success = run_experiment(paper, args.max_iterations, args.dry_run)

            if success:
                progress["completed"].append(paper)
                completed_count += 1
            else:
                progress["failed"].append(paper)
                failed_count += 1

            # Save progress after each paper
            save_progress(progress_file, progress)

            # Print summary
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            remaining = len(papers) - i
            avg_time = elapsed / i
            est_remaining = remaining * avg_time

            print(f"\nProgress: {i}/{len(papers)} papers")
            print(f"Completed: {completed_count}, Failed: {failed_count}")
            print(f"Elapsed: {elapsed:.1f}h, Remaining: {est_remaining:.1f}h")

    except KeyboardInterrupt:
        print("\n\n⚠️  Batch run interrupted by user")
        print("Progress has been saved. You can resume with:")
        print(f"  python run_batch_experiments.py --skip-completed --progress-file {progress_file}")
        sys.exit(1)

    # Final summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 3600

    print(f"\n{'='*60}")
    print(f"BATCH RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Total papers processed: {len(papers)}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.1f} hours ({total_time/24:.1f} days)")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if progress["failed"]:
        print("Failed papers:")
        for paper in progress["failed"]:
            print(f"  - {paper}")
        print()

    print(f"Progress saved to: {progress_file}")


if __name__ == "__main__":
    main()
