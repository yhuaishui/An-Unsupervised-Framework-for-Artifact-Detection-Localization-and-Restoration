import json
import os
import argparse
import core.praser as Praser

def manage_path(dir):
    base_path = os.path.dirname(os.path.dirname(__file__))
    return  os.path.normpath(os.path.join(base_path, dir))

def load_config(file_path):
    json_str = ""
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split('//')[0] + '\n'
            json_str += line
    config = json.loads(json_str)
    return config

def parse(config_path):
    config = load_config(config_path)
    
    config["restoration_config_path"] = manage_path(config["pipeline"]["restoration_model"]["config_path"])
    config["restoration_model_path"] = config["pipeline"]["restoration_model"]["model_path"]

    args = argparse.Namespace(config=config["restoration_config_path"], 
                              phase="test",
                              batch=None,
                              gpu_ids=None,
                              debug=False,
                              save_exp_log=False
                             )
    opt = Praser.parse(args)
    config["config_restoration"] = opt

    return config


