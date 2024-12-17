from flask import Flask, send_file
from flask_cors import CORS, cross_origin
from pathlib import Path
import pandas as pd
from . import tools

def get_fnames(csv_path, data_folder, species_level):
    """Returns full paths to images based on csv_path and data_folder"""
    df = pd.read_csv(csv_path, sep=";", encoding="ISO-8859-1")
    fpaths = tools.load_fpaths(df, data_folder, species_level)
    fnames = [fpath.name for fpath in fpaths]
    fpaths = [str(fpath) for fpath in fpaths]
    return fpaths, fnames

class Server():
    def __init__(self, csv_path, data_folder, species_level):
        fpaths, fnames = get_fnames(csv_path, data_folder, species_level)
        self.url_map = dict(zip(fnames, fpaths))
    
    def register_routes(self, app):
        @app.route("/<url>")
        @cross_origin()
        def serve_file(url):
            if url in self.url_map:
                filepath = self.url_map[url]
                return send_file(filepath)
            else:
                return f"File not found for url: {url}"

def run_serve(args):
    app = Flask(__name__)
    cors = CORS(app)
    server = Server(args.csv_path, args.data_folder, args.species_level)
    server.register_routes(app)
    app.run(debug=args.debug, port=args.port, host=args.host)
