import random
from queue import Queue

import carla
import cv2
import numpy as np


class CarlaBridge:
    def __init__(self, config: dict):
        self.__initialize_client(config)

        self.__initialize_actors(config)

        self.__spawn_vehicles()

    def __initialize_client(self, config: dict) -> None:
        """
        Initialize the carla client. Must have a carla server running on localhost
        :param config:
        :return:
        """
        self.__client: carla.Client = carla.Client("localhost", 2000)
        self.__client.set_timeout(5.0)
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
        location, rotation = self.__choose_vehicle_location()
        transform: carla.Transform = carla.Transform(location, rotation)

        # Spawn tesla with its camera
        self.__tesla_actor = self.__world.spawn_actor(self.__tesla, transform)
        self.__actors.append(self.__tesla_actor)

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.__cam_sensor = self.__world.spawn_actor(self.__cam, transform, attach_to=self.__tesla_actor)
        self.__cam_sensor.listen(lambda data: self.__cam_cb(data))
        self.cam_queue: Queue = Queue()
        self.__actors.append(self.__cam_sensor)

        # Spawn toyota in front of tesla
        transform = carla.Transform(carla.Location(location.x, location.y-15, location.z), rotation)
        self.__toyota_actor = self.__world.spawn_actor(self.__red_toyota, transform, attach_to=self.__tesla_actor)
        self.__actors.append(self.__toyota_actor)

        # Set autopilot for both vehicles
        self.__tesla_actor.set_autopilot(True, self.__tm_port)
        self.__tm.auto_lane_change(self.__tesla_actor, False)

        self.__toyota_actor.set_autopilot(True, self.__tm_port)
        self.__tm.auto_lane_change(self.__toyota_actor, False)

        self.__tm.set_desired_speed(self.__tesla_actor, 37)
        self.__tm.set_desired_speed(self.__toyota_actor, 30)

    @staticmethod
    def __choose_vehicle_location() -> tuple[carla.Location, carla.Rotation]:
        """
        :return tuple containing the Location and Rotation:
        """
        locations = [
            (carla.Location(x=-490.6, y=256.8, z=0.281942),
             carla.Rotation(pitch=0.000000, yaw=-90.0, roll=0.000000))
        ]

        location, rotation = random.choice(locations)
        return location, rotation

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

    def get_data(self) -> tuple[np.ndarray, carla.VehicleControl]:
        self.__world.tick()

        last_control: carla.VehicleControl = self.__tesla_actor.get_control()
        last_image: np.ndarray = self.cam_queue.get(True, 1.0)

        return last_control, last_image

