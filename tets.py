import gzip
import shutil

input_file = "NASA_access_log_Jul95.gz"
output_file = "NASA_access_log_Jul95"

with gzip.open(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Done!")
