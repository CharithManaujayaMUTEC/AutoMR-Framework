# automr/dashboard/config_manager.py

import json


def save_config(config, path):

    data = {
        "selected_mrs": config.selected_mrs,
        "mr_ranges": config.mr_ranges,
        "frame_skip": config.frame_skip,
        "live_intensity": config.live_intensity
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_config(config, path):

    with open(path, "r") as f:
        data = json.load(f)

    config.selected_mrs   = data["selected_mrs"]
    config.mr_ranges      = data["mr_ranges"]
    config.frame_skip     = data["frame_skip"]
    config.live_intensity = data.get("live_intensity", 50)