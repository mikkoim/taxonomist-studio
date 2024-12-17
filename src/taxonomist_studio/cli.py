
import argparse
import logging
import subprocess

from .ppp import run_ppp
from .mdma import run_mdma
from .segment import run_segment
from .embedder import run_embedder
from .serve import run_serve
from .labelstudio import run_labelstudio

from .dataset import scan_biodiscover_dataset
from .dataset import st_explore

logger = logging.getLogger(__name__)

def dataset(args=None):
    if args.command == "scan":
        scan_biodiscover_dataset.scan(args)
    elif args.command == "explore":
        subprocess.run(["streamlit", "run", 
                        st_explore.__file__,
                        "--server.maxUploadSize", "5000"], check=True)
    else:
        print(f"Invalid command {args.command} for dataset")

def main(args=None):

    parser = argparse.ArgumentParser(description="BioDiscover Studio is a suite for functions related to the BioDiscover imaging device") 
    subparsers = parser.add_subparsers(help="Available subcommands")

    parser_dataset = subparsers.add_parser("dataset", help="Dataset related functions")
    parser_dataset.add_argument("command", type=str)
    parser_dataset = scan_biodiscover_dataset.add_args(parser_dataset)
    parser_dataset.set_defaults(func=dataset)

    parser_mdma = subparsers.add_parser("mdma", help="Run the MetaDataManagingApplication")
    parser_mdma.add_argument("--debug", action="store_true")
    parser_mdma.add_argument("--version", type=str, default="latest")
    parser_mdma.set_defaults(func=run_mdma)

    parser_ppp = subparsers.add_parser("ppp", help="Run the PlatePositionPaster")
    parser_ppp.set_defaults(func=run_ppp)

    parser_segment = subparsers.add_parser("segment", help="Run the segmentation")
    parser_segment.add_argument("--data_folder", type=str, required=True)
    parser_segment.add_argument("--csv_path", type=str, required=True)
    parser_segment.add_argument("--out_folder", type=str, default=".")
    parser_segment.add_argument("--out_prefix", type=str, default="data")
    parser_segment.add_argument("--head", type=int)
    parser_segment.add_argument("--species_level", action="store_true")
    parser_segment.set_defaults(func=run_segment)

    parser_serve_dataset = subparsers.add_parser("serve-dataset", help="Serve the dataset")
    parser_serve_dataset.add_argument("--data_folder", type=str, required=True)
    parser_serve_dataset.add_argument("--csv_path", type=str, default=5000)
    parser_serve_dataset.add_argument("--port", type=int, default=5000)
    parser_serve_dataset.add_argument("--host", type=str, default="127.0.0.1")
    parser_serve_dataset.add_argument("--debug", action="store_true")
    parser_serve_dataset.add_argument("--species_level", action="store_true")
    parser_serve_dataset.set_defaults(func=run_serve)

    parser_embedder = subparsers.add_parser("embedder", help="Run the embedder")
    parser_embedder.add_argument("--data_folder", type=str, required=True)
    parser_embedder.add_argument("--csv_path", type=str, required=True)
    parser_embedder.add_argument("--timm_model", type=str, default='mobilenetv3_small_075.lamb_in1k')
    parser_embedder.add_argument("--batch_size", type=int, default=64)
    parser_embedder.add_argument("--out_folder", type=str, default=".")
    parser_embedder.add_argument("--out_fname", type=str, default="data")
    parser_embedder.add_argument("--head", type=int)
    parser_embedder.add_argument("--species_level", action="store_true")
    parser_embedder.set_defaults(func=run_embedder)

    parser_labelstudio = subparsers.add_parser("labelstudio", help="Run the Label Studio")
    parser_labelstudio.add_argument("--data_folder", type=str, required=True)
    parser_labelstudio.add_argument("--csv_path", type=str, default=5000)
    parser_labelstudio.add_argument("--port", type=int, default=5000)
    parser_labelstudio.add_argument("--host", type=str, default="127.0.0.1")
    parser_labelstudio.add_argument("--debug", action="store_true")
    parser_labelstudio.add_argument("--species_level", action="store_true")
    parser_labelstudio.set_defaults(func=run_labelstudio)

    args = parser.parse_args()
    
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()