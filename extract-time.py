import os


def extract_best_timing(file_path):
    best_timing = None
    with open(file_path, 'r') as file:
        for line in file:
            if "Time elapsed" in line:
                timing_str = line.split(":")[-1].strip()
                timing = int(timing_str.split()[0])
                best_timing = timing
                break
    return best_timing


current_directory = os.getcwd()
output_files = [file for file in os.listdir(current_directory) if file.endswith("output.txt")]
timings_dict = {}

for file_name in output_files:
    file_path = os.path.join(current_directory, file_name)
    best_timing = extract_best_timing(file_path)
    if best_timing is not None:
        timings_dict[file_name] = best_timing

sorted_timings_dict = dict(sorted(timings_dict.items()))
with open("times.csv", "a") as output_file:
    output_line = ",".join(str(timing) for timing in sorted_timings_dict.values())
    output_file.write(output_line + "\n")

for file_name in output_files:
    os.remove(os.path.join(current_directory, file_name))
