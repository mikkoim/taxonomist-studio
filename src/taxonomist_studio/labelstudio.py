import os
import subprocess
from subprocess import Popen

def run_labelstudio(args):
    os.environ["EXPERIMENTAL_FEATURES"] = '1'

    command = " ".join(["taxonomist-studio", "serve-dataset",
           "--data_folder", str(args.data_folder),
           "--csv_path", str(args.csv_path),
           "--port", str(args.port),
           "--host", str(args.host)])
    if args.debug:
        command += " --debug"
    if args.species_level:
        command += " --species_level"
    print(command)
    Popen(command)

    subprocess.run(["label-studio"])