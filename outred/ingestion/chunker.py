# outred/ingestion/chunker.py

import logging
import os
from typing import Iterator

import polars as pl

logger = logging.getLogger(__name__)


class CSVError(ValueError):
    """Raised when a CSV file cannot be opened or parsed.

    Using ValueError means FastAPI's existing ``except ValueError`` handler
    in server.py will catch it and return a 400 response automatically,
    without any server-side changes needed.
    """


def validate_file(file_path: str):
    # Validates if the file exists, is readable, and is a CSV.
    if not os.path.exists(file_path):
        msg = f"File not found: '{file_path}'"
        logger.error(msg)
        raise CSVError(msg)

    if not os.access(file_path, os.R_OK):
        msg = f"File is not readable: '{file_path}'"
        logger.error(msg)
        raise CSVError(msg)

    if not file_path.lower().endswith(".csv"):
        msg = f"Only CSV files are supported. Got: '{file_path}'"
        logger.error(msg)
        raise CSVError(msg)

    if os.path.getsize(file_path) == 0:
        msg = f"File is empty: '{file_path}'"
        logger.error(msg)
        raise CSVError(msg)


def count_csv_data_rows(file_path: str) -> int:
    """
    Cheaply count the number of DATA rows (excluding the header) in a CSV
    file, without parsing or typing any of it.

    This exists because the smart router needs the TRUE row count of the
    full file to make routing decisions (e.g. "rows > 1,000,000 -> HBOS"),
    but profiling only reads a small sample (e.g. 1,000 rows) for speed.
    Using the sample's row count for that decision is a bug: it makes the
    1,000,000-row threshold impossible to ever reach, since the sample is
    capped well below it.

    This counts raw newline bytes in 1MB buffered reads -- no CSV parsing,
    no dtype inference, no polars overhead. On a 500K-row file this runs in
    ~20ms, against ~200ms-1.2s for any polars-based row count (read_csv or
    scan_csv().select(pl.len())), because those still parse/type every
    field. For huge files (GBs), this stays cheap because it never holds
    more than one 1MB buffer in memory at a time.

    Handles the edge case where the file does not end with a trailing
    newline (common with some CSV writers) -- without this, the last row
    would silently be undercounted.
    """
    count = 0
    last_byte = b""
    with open(file_path, "rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            count += buf.count(b"\n")
            last_byte = buf[-1:]

    if last_byte and last_byte != b"\n":
        # Final line has no trailing newline but still contains data.
        count += 1

    # Subtract 1 for the header row. Floor at 0 for an empty/header-only file.
    return max(0, count - 1)


def _open_batched_reader(file_path: str, chunk_size: int):
    """
    Opens a Polars batched CSV reader with utf8-lossy encoding.

    Returns the reader object. Raises on failure  - the caller is expected
    to catch and present a user-friendly error directing them to
    ``--validate-only``.
    """
    return pl.read_csv_batched(
        file_path,
        batch_size=chunk_size,
        infer_schema_length=0,
        encoding='utf8-lossy',
    )


def stream_csv(file_path: str, chunk_size: int = 100_000) -> Iterator[pl.DataFrame]:
    """
    Streams a massive CSV file in memory-safe chunks using Polars.
    Yields one DataFrame at a time, ensuring RAM usage stays flat.

    If the file has structural issues (malformed quoting, broken encoding),
    raises CSVError (a ValueError subclass) with a clear message so the
    server can return a 400 response instead of crashing the process.
    """
    validate_file(file_path)

    logger.info("Starting to stream %s in chunks of %s rows...", file_path, f"{chunk_size:,}")

    try:
        reader = _open_batched_reader(file_path, chunk_size)
    except CSVError:
        raise
    except Exception as e:
        msg = (
            f"Failed to open CSV for reading: {str(e)[:300]}. "
            f"Your CSV may contain malformed quoting or invalid encoding."
        )
        logger.error(msg)
        raise CSVError(msg) from e

    chunk_count = 0

    while True:
        try:
            batches = reader.next_batches(1)
        except Exception as e:
            msg = (
                f"CSV parsing failed at chunk {chunk_count + 1}: {str(e)[:300]}. "
                f"Your CSV contains structural issues that prevent reliable parsing."
            )
            logger.error(msg)
            raise CSVError(msg) from e

        if not batches:
            break

        for df in batches:
            if not isinstance(df, pl.DataFrame):
                continue
            if len(df) == 0:
                continue

            # Warn if chunk has too many nulls (>50% of values)
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.null_count().sum_horizontal()[0]
            null_pct = (null_cells / total_cells) * 100 if total_cells > 0 else 0
            if null_pct > 50:
                logger.warning("Chunk %d is %.1f%% null -- results may be unreliable.", chunk_count + 1, null_pct)

            chunk_count += 1
            yield df

    if chunk_count == 0:
        msg = "CSV file has no data rows."
        logger.error(msg)
        raise CSVError(msg)

    logger.info("Finished streaming. Processed %d chunks.", chunk_count)