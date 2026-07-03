import csv
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Test: can Python's csv module read the malformed file?
with open('basic_habitation_2009.csv', 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f'Header ({len(header)} cols): {header[:5]}...')
    
    count = 0
    errors = 0
    for row in reader:
        count += 1
        if len(row) != len(header):
            errors += 1
            if errors <= 5:
                print(f'  Row {count}: got {len(row)} fields (expected {len(header)})')
                print(f'    Content: {row[:3]}...')
    
    print(f'Total rows: {count}')
    print(f'Rows with wrong field count: {errors}')
