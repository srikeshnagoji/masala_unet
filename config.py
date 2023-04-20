import yaml
from pathlib import Path

from munch import Munch

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    # config = _C.clone()
    # update_config(config, args)
    
    conf = yaml.safe_load(Path(args.cfg).read_text())
    return Munch.fromDict(conf)

