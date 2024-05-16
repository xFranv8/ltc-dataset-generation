import carla
import cv2

import os
import time

import numpy as np

from CarlaBridge import CarlaBridge


class DatasetGenerator:
    def __init__(self, output_dir: str, bridge: CarlaBridge):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.__output_dir = output_dir
        self.__bridge = bridge

    def generate_dataset(self) -> None:
        controls: list[carla.VehicleControl] = []

        t0: float = time.time()
        t1: float = time.time()
        delta: float = t1 - t0
        i: int = 0

        init_noise_time: float = 0.0
        noise: list[float] = [-0.005, 0.005]

        while delta < 0.2 * 60:
            if round(delta % 25) == 0:
                self.__bridge.noise = noise[np.random.randint(0, 2)]
                init_noise_time = time.time()
            else:
                if (init_noise_time != 0.0) and (time.time() - init_noise_time > 0.75):
                    self.__bridge.noise = None

            image, control = self.__bridge.get_data()

            controls.append(control)

            # cv2.imshow("", image)
            # cv2.waitKey(1)
            cv2.imwrite(f"{self.__output_dir}/{i}.png", image)

            if self.__bridge.collision:
                break

            i += 1

            t1 = time.time()
            delta = t1 - t0

        with open(f"{self.__output_dir}/data_out.csv", 'w') as f:
            f.write("throttle, steer, brake\n")
            for control in controls:
                throttle, steer, brake = control.throttle, control.steer, control.brake
                line = ','.join(map(str, [throttle, steer, brake]))

                f.write(line + '\n')

    def destroy_actors(self):
        self.__bridge.destroy_actors()
