import yaml

global cfg
if 'cfg' not in globals():
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

