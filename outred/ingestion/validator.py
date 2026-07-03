# outred/ingestion/validator.py
# CSV structural validation  - checks a file for issues that would prevent
# reliable parsing by the outlier detection pipeline.

import csv
import os
import sys
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationIssue:
    """A single structural issue found on one line of a CSV file."""
    line_number: int
    issue_type: str
    description: str
    raw_line: str


@dataclass
class ValidationResult:
    """Aggregate result of validating an entire CSV file."""
    is_valid: bool
    total_lines: int
    issues: List[ValidationIssue] = field(default_factory=list)


def validate_csv(
    file_path: str,
    output_dir: str = "results",
    write_reports: bool = True,
    max_lines: int = None,
) -> ValidationResult:
    """
    Scan every line of a CSV file for structural problems that would cause
    Polars (or any strict RFC 4180 parser) to fail.

    Checks performed per line:
      1. Invalid UTF-8 byte sequences
      2. NULL bytes
      3. Odd number of quote characters (unbalanced quoting / multiline records)
      4. Doubled-quote sequences ("") outside of valid RFC 4180 escaping
      5. Wrong field count after csv.reader parsing
      6. Stray literal quote characters inside parsed field values

    Args:
        file_path:     Path to the CSV file.
        output_dir:    Where to write report files (only used when write_reports=True).
        write_reports: If False, skip writing validation_report.txt/.csv to disk.
                       Use False when calling from the web server (ephemeral mode).
        max_lines:     Stop scanning after this many lines. None = scan everything.
                       Useful for large uploads where a quick scan is sufficient.

    Returns a ValidationResult with all issues collected.
    """
    issues: List[ValidationIssue] = []
    total_lines = 0
    header_col_count = None

    with open(file_path, "rb") as f:
        for lineno, raw in enumerate(f, 1):
            if max_lines is not None and lineno > max_lines:
                break
            total_lines += 1
            line_issues: List[str] = []

            # --- Check 1: UTF-8 validity ------------------------------------
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                issues.append(ValidationIssue(
                    line_number=lineno,
                    issue_type="Invalid UTF-8",
                    description=str(e),
                    raw_line=raw.decode("utf-8", errors="replace").rstrip(),
                ))
                continue

            line_stripped = line.rstrip("\r\n")

            # --- Check 2: NULL bytes ----------------------------------------
            if "\x00" in line:
                line_issues.append("NULL byte in line")

            # --- Check 3: Odd quote count (unbalanced quoting) --------------
            quote_count = line.count('"')
            if quote_count % 2:
                line_issues.append(
                    f"Odd quote count ({quote_count}) -- likely a multiline "
                    f"record or unclosed quoted field"
                )

            # --- Check 4: Stray "" -----------------------------------------
            if '""' in line:
                idx = line.find('""')
                # A valid "" is an escaped quote *inside* a quoted field,
                # e.g. "He said ""hello""".  Those are always preceded by
                # another " (the field-opening quote or a prior escape).
                # A *broken* "" at a field boundary (e.g. ,"" TEXT",) has a
                # comma or start-of-line before it.
                if idx == 0 or line[idx - 1] != '"':
                    line_issues.append('Contains "" -- possible broken quoted field')

            # --- Check 5 & 6: Parse with csv.reader -------------------------
            try:
                row = next(csv.reader([line]))
            except Exception as e:
                line_issues.append(f"csv.reader failed: {e}")
                row = None

            if row is not None:
                # Determine expected column count from header (line 1)
                if lineno == 1:
                    header_col_count = len(row)
                elif header_col_count is not None and len(row) != header_col_count:
                    line_issues.append(
                        f"Wrong field count: got {len(row)}, "
                        f"expected {header_col_count}"
                    )

                # Check for stray literal quotes in parsed values
                for i, value in enumerate(row):
                    if value.startswith('"'):
                        line_issues.append(
                            f'Field {i} starts with literal "')
                    if value.endswith('"'):
                        line_issues.append(
                            f'Field {i} ends with literal "')
                    if '"' in value and not (
                        value.startswith('"') and value.endswith('"')
                    ):
                        line_issues.append(
                            f'Field {i} contains stray "')

            # Collect issues for this line
            for desc in line_issues:
                issues.append(ValidationIssue(
                    line_number=lineno,
                    issue_type=_classify_issue(desc),
                    description=desc,
                    raw_line=line_stripped,
                ))

    is_valid = len(issues) == 0
    result = ValidationResult(
        is_valid=is_valid,
        total_lines=total_lines,
        issues=issues,
    )

    # Write report files (CLI mode only — skip in server/ephemeral mode)
    if write_reports:
        os.makedirs(output_dir, exist_ok=True)
        _write_text_report(result, file_path, output_dir)
        _write_csv_report(result, output_dir)

    return result


def _classify_issue(description: str) -> str:
    """Map a description string to a short issue type label."""
    desc_lower = description.lower()
    if "utf-8" in desc_lower or "utf8" in desc_lower:
        return "Invalid encoding"
    if "null byte" in desc_lower:
        return "NULL byte"
    if "odd quote" in desc_lower:
        return "Multiline record"
    if '""' in description:
        return "Broken quoted field"
    if "field count" in desc_lower:
        return "Wrong field count"
    if "csv.reader" in desc_lower:
        return "Parse error"
    if "stray" in desc_lower or "starts with" in desc_lower or "ends with" in desc_lower:
        return "Invalid quoting"
    return "Unknown"


def _write_text_report(
    result: ValidationResult, file_path: str, output_dir: str
) -> str:
    """Write the human-readable validation_report.txt."""
    report_path = os.path.join(output_dir, "validation_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  OUTRED -- CSV Validation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"  File   : {file_path}\n")
        f.write(f"  Lines  : {result.total_lines:,}\n")
        f.write(f"  Status : {'PASSED' if result.is_valid else 'FAILED'}\n")

        if not result.is_valid:
            # Count unique lines with issues
            unique_lines = len({iss.line_number for iss in result.issues})
            f.write(f"  Issues : {len(result.issues):,} issues on "
                    f"{unique_lines:,} lines\n")

        f.write("\n" + "-" * 60 + "\n\n")

        if result.is_valid:
            f.write("  No structural issues found. File is ready for OUTRED.\n")
        else:
            # Group issues by line number for readable output
            current_line = None
            for issue in result.issues:
                if issue.line_number != current_line:
                    if current_line is not None:
                        # Print the raw line content for the previous group
                        f.write("\n")
                    current_line = issue.line_number
                    f.write(f"[Line {issue.line_number}]\n")

                f.write(f"  Type   : {issue.issue_type}\n")
                f.write(f"  Reason : {issue.description}\n")

            # Final raw line
            f.write("\n")

            f.write("-" * 60 + "\n\n")
            f.write("  CSV validation failed.\n")
            f.write("  Please correct the reported rows and rerun OUTRED.\n")

        f.write("\n" + "=" * 60 + "\n")

    return report_path


def _write_csv_report(result: ValidationResult, output_dir: str) -> str:
    """Write the machine-readable validation_report.csv."""
    report_path = os.path.join(output_dir, "validation_report.csv")

    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["line", "type", "description", "raw_content"])
        for issue in result.issues:
            writer.writerow([
                issue.line_number,
                issue.issue_type,
                issue.description,
                issue.raw_line,
            ])

    return report_path
