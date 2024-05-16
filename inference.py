import carla
import torch.cuda

from ltcs.TrainingModule import TrainingModule
from CarlaBridge import CarlaBridge


def main(args):
    PATH: str = "ltcs/ltc-carla-2024-04-22 14:31:11.050334-finetunned.ckpt"

    print(torch.cuda.is_available())

    model = TrainingModule.load_from_checkpoint(PATH)

    model.model = model.model.eval()
    config: dict = {
        "town": "Town04",
        "weather": carla.WeatherParameters(cloudiness=70.0, precipitation=0.0, sun_altitude_angle=70.0),
        "im_width": 640,
        "im_height": 480
    }
    carla_bridge: CarlaBridge = CarlaBridge(config, model)

    while True:
        carla_bridge.run_inference()


if __name__ == '__main__':
    main(None)
