# main.py
# CLI entry point for Outred -- outlier detection pipeline for CSV files.
# Also serves as the launcher for the web UI via --serve.

import argparse
import json
import os
import sys

from outred.config import OutredConfig, VALID_ALGORITHMS, VALID_SCALING, VALID_IMPUTE, VALID_CAT_FREQ_MODES


def parse_args():
    parser = argparse.ArgumentParser(
        prog="outred",
        description="Outlier detection pipeline for large tabular CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i data.csv                        # auto-select algorithm
  python main.py -i data.csv --algorithm ensemble   # force ensemble mode
  python main.py -i data.csv -a hbos -n 0.10        # HBOS with 10%% contamination
  python main.py -i data.csv --explain              # include SHAP explanations
  python main.py --serve                            # launch web UI

Advanced tuning (for diverse datasets):
  python main.py -i habitation.csv --cat-max-cardinality 0.02 --cat-threshold 0.005
  python main.py -i data.csv --sample-rows 5000 --id-cardinality 0.30
  python main.py -i data.csv --cat-freq-mode single-pass  # faster on small files
  python main.py -i data.csv --global-sample-rows 100000  # larger global model sample
        """,
    )

    # --- Mode ---------------------------------------------------------------
    parser.add_argument(
        "--serve", action="store_true",
        help="Launch the web UI server instead of running CLI detection.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate CSV structure without running detection. Produces a "
             "validation report showing any malformed rows.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host for the web server. (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the web server. (default: 8000)",
    )

    # --- I/O ----------------------------------------------------------------
    parser.add_argument(
        "--input", "-i",
        help="Path to input CSV file. (required for CLI mode)",
    )
    parser.add_argument(
        "--output", "-o", default="results/output.parquet",
        help="Path to output Parquet file. (default: results/output.parquet)",
    )

    # --- Algorithm ----------------------------------------------------------
    parser.add_argument(
        "--algorithm", "-a", default="auto",
        choices=VALID_ALGORITHMS,
        help="Detection algorithm. 'auto' lets the smart router choose. (default: auto)",
    )
    parser.add_argument(
        "--contamination", "-n", type=float, default=0.05,
        help="Expected outlier proportion, 0.001-0.20. (default: 0.05)",
    )

    # --- Preprocessing ------------------------------------------------------
    parser.add_argument(
        "--scaling", "-s", default="robust",
        choices=VALID_SCALING,
        help="Feature scaling method. (default: robust)",
    )
    parser.add_argument(
        "--impute", default="median",
        choices=VALID_IMPUTE,
        help="Null value handling strategy. (default: median)",
    )

    # --- Processing ---------------------------------------------------------
    parser.add_argument(
        "--chunk-size", "-c", type=int, default=100_000,
        help="Rows per chunk for streaming. (default: 100,000)",
    )

    # --- Explainability -----------------------------------------------------
    parser.add_argument(
        "--explain", "-e", action="store_true",
        help="Compute SHAP feature contributions for flagged outliers.",
    )

    # --- Advanced Settings --------------------------------------------------
    advanced = parser.add_argument_group(
        "Advanced Settings",
        "Tune categorical detection, profiling, and routing thresholds for diverse datasets.",
    )
    advanced.add_argument(
        "--cat-threshold", type=float, default=0.01,
        help="Absolute frequency floor for rare-category detection. Used as a cap "
             "alongside the adaptive threshold. (default: 0.01 = 1%%)",
    )
    advanced.add_argument(
        "--cat-max-cardinality", type=float, default=0.10,
        help="Max unique/total ratio to treat a column as categorical. Columns with "
             "higher cardinality (names, addresses) are skipped. (default: 0.10 = 10%%)",
    )
    advanced.add_argument(
        "--cat-min-cardinality", type=int, default=10,
        help="Minimum unique value floor for the cardinality check. (default: 10)",
    )
    advanced.add_argument(
        "--cat-freq-mode", default="two-pass",
        choices=VALID_CAT_FREQ_MODES,
        help="How to compute categorical value frequencies for rare-value detection. "
             "'two-pass' (default) streams the full file for exact global counts -- "
             "more accurate on large or sorted files, but adds one extra I/O pass. "
             "'single-pass' uses the profiling sample as a proxy -- faster, recommended "
             "for datasets < 100K rows where the sample already covers most values.",
    )
    advanced.add_argument(
        "--id-cardinality", type=float, default=0.50,
        help="ID detection cardinality ratio in profiling sample. String columns with "
             "unique/sample > this ratio are auto-excluded as IDs. (default: 0.50 = 50%%)",
    )
    advanced.add_argument(
        "--numeric-cast", type=float, default=0.80,
        help="Fraction of parseable values to cast a string column to numeric. "
             "Lower = more aggressive casting. (default: 0.80 = 80%%)",
    )
    advanced.add_argument(
        "--sample-rows", type=int, default=1000,
        help="Number of rows to sample for profiling. Larger = more accurate cardinality "
             "estimates but slower startup. (default: 1000)",
    )
    advanced.add_argument(
        "--global-sample-rows", type=int, default=50_000,
        help="Number of rows sampled uniformly from the file to fit the global numeric "
             "model (Option B architecture). Larger = more representative model but "
             "higher RAM usage during fitting. (default: 50,000)",
    )
    advanced.add_argument(
        "--route-incremental-mb", type=float, default=500.0,
        help="File size (MB) above which the incremental engine is auto-selected. (default: 500)",
    )
    advanced.add_argument(
        "--route-hbos-rows", type=int, default=1_000_000,
        help="Row count above which HBOS is auto-selected. (default: 1,000,000)",
    )
    advanced.add_argument(
        "--route-high-dims", type=int, default=50,
        help="Numeric dimension count above which IForest is preferred. (default: 50)",
    )
    advanced.add_argument(
        "--route-skewness", type=float, default=5.0,
        help="Average |skewness| above which LOF is auto-selected. (default: 5.0)",
    )

    return parser.parse_args()


def run_validate(input_path: str, output_dir: str = "results"):
    """Run CSV validation only -- no detection."""
    from outred.ingestion.validator import validate_csv

    print()
    print("  +------------------------------------------+")
    print("  |         OUTRED -- CSV Validator          |")
    print("  +------------------------------------------+")
    print()
    print(f"  Input : {input_path}")
    print(f"  Scanning for structural issues...")
    print()

    result = validate_csv(input_path, output_dir=output_dir)

    if result.is_valid:
        print(f"  Status : PASSED")
        print(f"  Lines  : {result.total_lines:,}")
        print(f"  No structural issues found. File is ready for OUTRED.")
    else:
        unique_lines = len({iss.line_number for iss in result.issues})
        print(f"  Status : FAILED")
        print(f"  Lines  : {result.total_lines:,}")
        print(f"  Issues : {len(result.issues):,} issues on {unique_lines:,} lines")
        print()
        print(f"  Reports written to:")
        print(f"    {output_dir}/validation_report.txt")
        print(f"    {output_dir}/validation_report.csv")
        print()
        print(f"  Please correct the reported rows and rerun OUTRED.")

    print()
    sys.exit(0 if result.is_valid else 1)


def run_cli(config: OutredConfig, input_path: str):
    """Run the CLI detection pipeline."""
    from outred.router.dispatcher import dispatch_batch
    from outred.reporter.aggregator import ReportAggregator

    print()
    print("  +------------------------------------------+")
    print("  |            OUTRED -- V1 Engine           |")
    print("  +------------------------------------------+")
    print()
    print(f"  Input           : {input_path}")
    print(f"  Output          : {config.output_path}")
    print(f"  Algorithm       : {config.algorithm}")
    print(f"  Contamination   : {config.contamination}")
    print(f"  Scaling         : {config.scaling}")
    print(f"  Imputation      : {config.impute}")
    print(f"  Chunk size      : {config.chunk_size:,}")
    print(f"  Cat freq mode   : {config.cat_freq_mode}")
    print(f"  Global sample   : {config.global_sample_rows:,} rows")
    print(f"  Explain         : {config.explain}")
    print()

    aggregator = ReportAggregator(output_path=config.output_path)

    all_explanations = []
    for result in dispatch_batch(file_path=input_path, config=config):
        aggregator.add_chunk(result.chunk)
        if result.explanations:
            all_explanations.extend(result.explanations)

    aggregator.finalize()

    if config.explain:
        if all_explanations:
            explanations_path = _explanations_sidecar_path(config.output_path)
            os.makedirs(os.path.dirname(explanations_path) or ".", exist_ok=True)
            with open(explanations_path, "w") as f:
                json.dump(all_explanations, f, indent=2)
            print(f"  Explanations : {len(all_explanations)} rows explained, "
                  f"saved to {explanations_path}")
        else:
            print("  Explanations : --explain was set but no explanations were "
                  "generated (either no outliers were flagged, or the selected "
                  "algorithm doesn't support explanation yet -- see console "
                  "output above for details).")


def _explanations_sidecar_path(output_path: str) -> str:
    """
    Derive a sidecar JSON path from the Parquet output path, e.g.
    'results/output.parquet' -> 'results/output.explanations.json'.
    """
    base, _ext = os.path.splitext(output_path)
    return f"{base}.explanations.json"


def run_server(host: str, port: int):
    """Launch the FastAPI web server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Run: pip install uvicorn")
        sys.exit(1)

    print()
    print("  +------------------------------------------+")
    print("  |         OUTRED -- Web UI Server          |")
    print("  +------------------------------------------+")
    print()
    print(f"  Starting server at http://{host}:{port}")
    print(f"  Press Ctrl+C to stop.")
    print()

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        h11_max_incomplete_event_size=1024 * 1024 * 1024,  # 1 GB — must match server.py MAX_UPLOAD_BYTES
    )


def main():
    args = parse_args()

    # Web server mode
    if args.serve:
        run_server(args.host, args.port)
        return

    # CLI mode -- input is required
    if not args.input:
        print("Error: --input is required in CLI mode. Use --serve for web UI.")
        sys.exit(1)

    # Validate-only mode
    if args.validate_only:
        output_dir = os.path.dirname(args.output) or "results"
        run_validate(args.input, output_dir=output_dir)
        return

    # Build config -- all advanced settings are wired through
    config = OutredConfig(
        algorithm=args.algorithm,
        contamination=args.contamination,
        scaling=args.scaling,
        impute=args.impute,
        chunk_size=args.chunk_size,
        output_path=args.output,
        explain=args.explain,
        # Advanced settings
        cat_rare_threshold=args.cat_threshold,
        cat_max_cardinality_ratio=args.cat_max_cardinality,
        cat_min_cardinality=args.cat_min_cardinality,
        cat_freq_mode=args.cat_freq_mode,
        profiler_id_cardinality_ratio=args.id_cardinality,
        numeric_cast_threshold=args.numeric_cast,
        profiler_sample_rows=args.sample_rows,
        global_sample_rows=args.global_sample_rows,
        route_incremental_size_mb=args.route_incremental_mb,
        route_hbos_row_threshold=args.route_hbos_rows,
        route_high_dim_threshold=args.route_high_dims,
        route_skewness_threshold=args.route_skewness,
    )

    try:
        config.validate()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    run_cli(config, args.input)


if __name__ == "__main__":
    main()