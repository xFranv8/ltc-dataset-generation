from CarlaBridge import CarlaBridge

import carla


def main():
    config: dict = {
        "town": "Town04",
        "weather": carla.WeatherParameters(cloudiness=70.0, precipitation=0.0, sun_altitude_angle=70.0),
        "im_width": 640,
        "im_height": 480
    }
    carla_bridge: CarlaBridge = CarlaBridge(config)

    while True:
        carla_bridge.run()


if __name__ == '__main__':
    main()
