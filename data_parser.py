import re
import os

def parse_data_file(file_path):
    data = {}

    # Extract metadata from filename
    filename = os.path.basename(file_path)
    match = re.match(r'data_(\d+)_(\d+)_(\d+)_(\d+)\.dat', filename)
    if match:
        data['file_number'] = int(match.group(1))
        data['num_workers'] = int(match.group(2))
        data['num_jobs'] = int(match.group(3))  # double-check against file content
        data['multi_skilling_level'] = int(match.group(4))
    else:
        raise ValueError(f"Filename '{filename}' doesn't match expected pattern.")

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    # Skip first 3 lines (comments)
    lines = lines[3:]

    # Line 2: Type
    type_match = re.search(r'Type\s*=\s*(\d+)', lines[0])
    if type_match:
        data['type'] = int(type_match.group(1))
    else:
        raise ValueError("Couldn't parse 'Type =' line.")

    # Next num_jobs lines: Job start and end times
    job_lines = lines[2:2 + data['num_jobs']]
    data['jobs'] = [tuple(map(int, line.split())) for line in job_lines]

    # Number of shifts (qualifications)
    shift_line_idx = 2 + data['num_jobs']
    num_qualifications = int(lines[shift_line_idx].split('=')[1].strip())
    data['num_qualifications'] = num_qualifications

    # Next lines: qualifications
    qualification_lines = lines[shift_line_idx + 1: shift_line_idx + 1 + num_qualifications]

    qualifications = {}
    for shift_idx, line in enumerate(qualification_lines):
        parts = line.replace(":", "").split()
        job_count = int(parts[0])  # Not strictly needed
        qualified_job_indices = list(map(int, parts[1:]))
        qualifications[shift_idx] = qualified_job_indices

    data['qualifications'] = qualifications

    return data