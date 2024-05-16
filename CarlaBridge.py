import time
import random
from queue import Queue

from simulator.PythonAPI.carla.agents.navigation.controller import VehiclePIDController

import carla
import cv2
import numpy as np
import torch


class CarlaBridge:
    def __init__(self, config: dict, model=None):
        self.__model = model

        self.__world = None
        self.__client = None
        self.__tm = None
        self.__tm_port = None

        self.noise = None

        self.__initialize_client(config)

        self.__initialize_actors(config)

        self.__spawn_vehicles()

        self.__wait_for_tick()

        self.count = 0

    def __initialize_client(self, config: dict) -> None:
        """
        Initialize the carla client. Must have a carla server running on localhost
        :param config:
        :return:
        """
        self.__client: carla.Client = carla.Client("localhost", 2000)
        self.__client.set_timeout(5.0)

        self.update_config(config)

    def __initialize_actors(self, config) -> None:
        """
        Initialize both vehicles models and the camera.
        :param config:
        :return:
        """
        self.__IM_WIDTH = config["im_width"]
        self.__IM_HEIGHT = config["im_height"]

        blueprint_library: carla.BlueprintLibrary = self.__world.get_blueprint_library()
        self.__tesla = blueprint_library.filter("model3")[0]
        self.__cam = blueprint_library.find("sensor.camera.rgb")
        self.__collision_sensor = blueprint_library.find('sensor.other.collision')
        self.collision = False

        self.__cam.set_attribute("image_size_x", f"{self.__IM_WIDTH}")
        self.__cam.set_attribute("image_size_y", f"{self.__IM_HEIGHT}")
        self.__cam.set_attribute("fov", "110")

        self.__red_toyota = blueprint_library.filter("prius")[0]

        self.__tesla_actor = None
        self.__toyota_actor = None
        self.__cam_sensor = None
        self.__actors: list = []

    def __spawn_vehicles(self) -> None:
        """
        Spawn one vehicle in front of each other and start collecting images.
        :return:
        """
        transform_tesla, transform_toyota = self.__choose_locations()

        # Spawn tesla with its camera
        self.__tesla_actor = self.__world.spawn_actor(self.__tesla, transform_tesla)
        self.__actors.append(self.__tesla_actor)

        transform_cam = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.__cam_sensor = self.__world.spawn_actor(self.__cam, transform_cam, attach_to=self.__tesla_actor)

        self.cam_queue: Queue = Queue()
        self.__cam_sensor.listen(lambda data: self.__cam_cb(data))
        self.__actors.append(self.__cam_sensor)

        transform_col = carla.Transform()
        collision_sensor = self.__world.spawn_actor(self.__collision_sensor, transform_col,
                                                    attach_to=self.__tesla_actor)
        collision_sensor.listen(lambda event: self.__collision_sensor_cb(event))
        self.__actors.append(collision_sensor)

        # Spawn toyota in front of tesla
        self.__toyota_actor: carla.Vehicle = self.__world.spawn_actor(self.__red_toyota, transform_toyota,
                                                                      attach_to=self.__tesla_actor)
        self.__actors.append(self.__toyota_actor)

        # Set controllers for both vehicles
        if self.__model is None:
            # self.__tesla_actor.set_autopilot(True, self.__tm_port)

            args_lateral: dict = {"K_P": 0.8, "K_D": 0.1, "K_I": 0.1}
            args_longitudinal: dict = {"K_P": 0.8, "K_D": 0.1, "K_I": 0.1}

            self.__pid: VehiclePIDController = VehiclePIDController(self.__tesla_actor, args_lateral, args_longitudinal)
            # self.__tm.auto_lane_change(self.__tesla_actor, False)
        else:
            self.__seq: torch.Tensor = torch.zeros(1, 64, 3, 480, 640)

        self.__toyota_actor.set_autopilot(True, self.__tm_port)
        # self.__tm.auto_lane_change(self.__toyota_actor, False)
        self.__tm.set_desired_speed(self.__toyota_actor, 30)

    def __cam_cb(self, data: carla.Image):
        """
        Process the image at every timestep
        :param data pure carla.Image:
        :return:
        """
        image = np.array(data.raw_data)
        image = image.reshape((data.height, data.width, 4))
        image = image[:, :, :3]

        self.cam_queue.put(image)

    def __collision_sensor_cb(self, event):
        self.collision = True

    def __wait_for_tick(self) -> None:
        ready: bool = False
        while not ready:
            try:
                self.__world.tick()
                image = self.cam_queue.get(True, 1.0)
                ready = True

            except Exception:
                time.sleep(1)

    def __choose_locations(self) -> tuple[carla.Transform, carla.Transform]:
        # Locations are paired by positions in the arrays

        locations_tesla = [
            (carla.Location(x=-490.6, y=256.8, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=-486.6, y=256.8, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=-494.6, y=256.8, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=412.5, y=-34.125988, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=408.5, y=-34.125988, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000))
        ]

        locations_toyota = [
            (carla.Location(x=-490.6, y=256.8 - 15, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=-490.6, y=256.8 - 15, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=-490.6, y=256.8 - 15, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=412.5, y=-44.43, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000)),

            (carla.Location(x=412.5, y=-44.43, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000))
        ]

        # i: int = np.random.randint(0, len(locations_tesla))
        map = self.__world.get_map()

        transform_tesla: carla.Transform = random.choice(self.__world.get_map().get_spawn_points())

        next_location: carla.Location = map.get_waypoint(transform_tesla.location).next(8.0)[0].transform.location

        waypoint_toyota: carla.Waypoint = map.get_waypoint(next_location)

        transform_toyota: carla.Transform = waypoint_toyota.transform
        transform_toyota.location.z = transform_tesla.location.z

        # transform_tesla: carla.Transform = carla.Transform(locations_tesla[i][0], locations_tesla[i][1])
        # transform_toyota: carla.Transform = carla.Transform(locations_toyota[i][0], locations_toyota[i][1])

        return transform_tesla, transform_toyota

    def calc_norm(self, vector: carla.Vector3D) -> float:
        return np.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)

    def get_data(self) -> tuple[np.ndarray, carla.VehicleControl]:
        self.__world.tick()

        toyota_location: carla.Location = self.__toyota_actor.get_location()
        tesla_location: carla.Location = self.__tesla_actor.get_location()
        distance: float = self.calc_norm(toyota_location - tesla_location)

        next_waypoint: carla.Waypoint = self.__world.get_map().get_waypoint(toyota_location)
        target_velocity: float = self.calc_norm(self.__toyota_actor.get_velocity()) * 3.6

        if distance > 15:
            target_velocity += 5

        elif distance < 5:
            target_velocity = max(0.0, target_velocity - 5)

        control: carla.VehicleControl = self.__pid.run_step(target_velocity, next_waypoint)

        if self.noise is not None:
            noisy_control: carla.VehicleControl = carla.VehicleControl()

            noisy_control.throttle = control.throttle
            noisy_control.brake = control.brake

            noisy_control.steer += self.noise

            self.__tesla_actor.apply_control(noisy_control)
        else:
            self.__tesla_actor.apply_control(control)

        last_image: np.ndarray = self.cam_queue.get(True, 1.0)

        return last_image, control

    def update_config(self, config: dict):
        self.__client.load_world(config["town"])

        self.__world = self.__client.get_world()
        settings = self.__world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05

        self.__tm = self.__client.get_trafficmanager(1212)
        self.__tm_port = self.__tm.get_port()
        self.__tm.set_global_distance_to_leading_vehicle(5)
        self.__tm.set_synchronous_mode(True)
        self.__world.apply_settings(settings)

        self.__client.reload_world(False)
        self.__world.set_weather(config["weather"])

    def run_inference(self):
        self.__world.tick()

        control: carla.VehicleControl = carla.VehicleControl()

        image: np.ndarray = self.cam_queue.get(True, 1.0)

        cv2.imshow("", image)
        cv2.waitKey(1)

        image = np.float32(image) / 255
        tensor = torch.tensor(image).permute(2, 0, 1)

        self.__seq[0, 0:-1, ...] = self.__seq[0, 1:, ...].clone()
        self.__seq[0, -1, ...] = tensor

        # last_control: carla.VehicleControl = self.__tesla_actor.get_control()

        self.count += 1
        if self.count > 64 * 5:
            with torch.no_grad():
                y_hat = self.__model(self.__seq).cpu().numpy()

            control.throttle = float(round(y_hat[0, -1, 0], 4))
            control.steer = float(round(y_hat[0, -1, 1], 4))
            control.brake = float(round(y_hat[0, -1, 2], 4))

            self.__tesla_actor.set_autopilot(False, self.__tm_port)
            self.__tesla_actor.apply_control(control)

    def destroy_actors(self) -> None:
        self.__client.apply_batch([carla.command.DestroyActor(x) for x in self.__actors])
