import csv

FILE = "basic_habitation_2009.csv"

with open(FILE, "rb") as f:
    for lineno, raw in enumerate(f, 1):
        issues = []

        # UTF-8 check
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            print(f"Line {lineno}: INVALID UTF-8 ({e})")
            continue

        # NULL bytes
        if "\x00" in line:
            issues.append("NULL byte")

        # Odd number of quotes
        if line.count('"') % 2:
            issues.append(f"Odd quote count ({line.count(chr(34))})")

        # Starts with double quote pair
        if '""' in line:
            idx = line.find('""')
            # Ignore valid escaped quotes like """ inside a field
            if idx == 0 or line[idx - 1] != '"':
                issues.append('Contains ""')

        # Mixed quotes
        if '\'"' in line or '"\'' in line:
            issues.append("Mixed single/double quotes")

        # Parse with csv.reader
        try:
            row = next(csv.reader([line]))
        except Exception as e:
            issues.append(f"csv.reader failed: {e}")
            row = None

        if row:
            for i, field in enumerate(row):
                if field.startswith('"'):
                    issues.append(f'Field {i} starts with literal "')

                if field.endswith('"'):
                    issues.append(f'Field {i} ends with literal "')

                if '"' in field and not (field.startswith('"') and field.endswith('"')):
                    issues.append(f'Field {i} contains stray "')

        if issues:
            print(f"\nLine {lineno}")
            for issue in sorted(set(issues)):
                print(f"  - {issue}")
            print(line.rstrip())