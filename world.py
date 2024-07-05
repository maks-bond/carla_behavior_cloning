import re
import carla
import random
import weakref
import math
import collections

from carla import ColorConverter as cc

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

import pygame

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

FIXED_DELTA_SECONDS = 0.05

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

def loc(x, y, z):
    return carla.Location(x, y, z)

def to_np(location):
    return np.array([location.x, location.y, location.z])

# It would be cool to switch to quaternions
def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R

def transform_to_rotation_matrix(transform):
    transform_matrix = transform.get_matrix()
    rotation_matrix = np.zeros((3, 3))
    for row_idx in range(3):
        rotation_matrix[row_idx] = transform_matrix[row_idx][:3]
    return rotation_matrix

def compute_lateral_distance2(source_point, target_point, source_rotation):
    rotation_matrix = euler_to_rotation_matrix(source_rotation.roll, source_rotation.pitch, source_rotation.yaw)
    target_relative_to_source = to_np(target_point-source_point)
    target_relative_to_source[2] = 0.0

    target_in_source = np.dot(rotation_matrix.T, target_relative_to_source)

    return target_in_source[1]

def compute_lateral_distance(source_point, target_point, transform):
    rotation_matrix = transform_to_rotation_matrix(transform)
    target_relative_to_source = to_np(target_point-source_point)
    target_relative_to_source[2] = 0.0

    target_in_source = np.dot(rotation_matrix.T, target_relative_to_source)

    return target_in_source[1]

# taken from Chat GPT.
def calculate_curvature(p1, p2, p3):
    # Extract coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate the side lengths of the triangle
    a = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    c = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    # Calculate the area of the triangle using Heron's formula
    s = (a + b + c) / 2  # Semi-perimeter
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Check if the points are collinear (area == 0)
    if area == 0:
        return 0.0  # Curvature is zero for collinear points
    
    # Calculate the circumradius
    circumradius = (a * b * c) / (4 * area)
    
    # Calculate the curvature (1 / circumradius)
    curvature = 1 / circumradius
    
    return curvature

# Now I need to get future points.
# How can I do that?

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0.0*bound_x, y=+0.0*bound_y, z=6.0*bound_z), carla.Rotation(yaw=0.0, pitch=-90.0, roll=0.0)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=1.0*bound_x, y=+0.0*bound_y, z=6.0*bound_z), carla.Rotation(yaw=0.0, pitch=-90.0, roll=0.0)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=1.0*bound_x, y=-1.0*bound_y, z=3.0*bound_z), carla.Rotation(yaw=0.0, pitch=-90.0, roll=0.0)), Attachment.Rigid)
                ]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0.0, z=0.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)] 

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class World(object):
    def __init__(self, carla_world, hud, traffic_manager, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self.traffic_manager = traffic_manager
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma

        # Player is created here
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

        self.VIZ_Z = 0.3

        self.setup_traffic_manager()

    def setup_traffic_manager(self):
        self.traffic_manager.set_synchronous_mode(True)

        path = []
        for i in range(5):
            path.append(carla.Location(6.0, 306.6, 0.5))
            path.append(carla.Location(192.8, 287.4, 0.5))
            path.append(carla.Location(182.6, 104.9, 0.5))
            path.append(carla.Location(-7.6, 129.2, 0.5))
        self.traffic_manager.set_path(self.player, path)
        self.traffic_manager.ignore_lights_percentage(self.player, 100.0)
        self.traffic_manager.set_desired_speed(self.player, 30.0)

    def get_distance_to_left_boundary(self, left_corner, rotation, left_boundary):
        transform = carla.Transform(left_corner, rotation)
        bnd_in_corner = transform.transform(carla.Location(x=left_boundary.x, y=left_boundary.y, z=left_boundary.z))

        return -bnd_in_corner.y
    
    def get_distance_to_right_boundary(self, right_corner, rotation, right_boundary):
        transform = carla.Transform(right_corner, rotation)
        bnd_in_corner = transform.transform(carla.Location(x=right_boundary.x, y=right_boundary.y, z=right_boundary.z))

        return bnd_in_corner.y

    def draw_box(self, location, rotation, color):
        BOX_WIDTH = 0.05
        THICKNESS = 0.01
        debug_location = carla.Location(location.x, location.y, location.z+self.VIZ_Z)
        box = carla.BoundingBox(debug_location, carla.Vector3D(BOX_WIDTH, BOX_WIDTH, BOX_WIDTH))
        self.world.debug.draw_box(box, rotation, thickness=THICKNESS, life_time=FIXED_DELTA_SECONDS+0.01, color=color)

    def draw_arrow(self, start_location, end_location, color, arrow_size=0.1):
        THICKNESS = 0.01
        debug_start_location = carla.Location(start_location.x, start_location.y, start_location.z+self.VIZ_Z)
        debug_end_location = carla.Location(end_location.x, end_location.y, end_location.z+self.VIZ_Z)
        self.world.debug.draw_arrow(debug_start_location, debug_end_location, thickness=THICKNESS, life_time=FIXED_DELTA_SECONDS+0.01, color=color, arrow_size=arrow_size)

    def draw_string(self, location, text, color):
        debug_location = carla.Location(location.x, location.y, location.z+self.VIZ_Z+0.02)
        self.world.debug.draw_string(debug_location, text, life_time=FIXED_DELTA_SECONDS+0.01, color=color)
    
    def min_distance_to_boundary(self, corners, boundary, transform, is_left):
        min_dist = float("inf")
        for corner in corners:
            lateral_dist = compute_lateral_distance(corner, boundary, transform)
            if is_left:
                lateral_dist = -lateral_dist
            
            min_dist = min(min_dist, lateral_dist)
        
        return min_dist

    # Perhaps I should build my own route, connecting waypoints.

    def compute_features(self):
        next_action = self.traffic_manager.get_next_action(self.player)
        print("Next action is: ", next_action)

        av = self.player
        location = self.player.get_location()
        transform = self.player.get_transform()
        rotation = transform.rotation

        print("Location x: ", location.x, ", locataion.y: ", location.y, ", location.z: ", location.z)
        print("rotation pitch: ", rotation.pitch, ", rotation.yaw: ", rotation.yaw, ", rotation.roll: ", rotation.roll)

        WAYPOINT_COLOR = carla.Color(255,0,0,100)
        CORNER_COLOR = carla.Color(0,0,255,100)
        BOUNDARY_COLOR = carla.Color(0,255,0,100)
        
        AXIS_X_COLOR = carla.Color(255, 5, 5, 100)
        AXIS_Y_COLOR = carla.Color(255, 105, 105, 100)
        AXIS_Z_COLOR = carla.Color(255, 205, 205, 100)

        map = self.world.get_map()
        # Waypoint represents closest lane sample to given location.
        waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_center = waypoint.transform.location
        rotation = waypoint.transform.rotation

        lane_width = waypoint.lane_width

        self.draw_box(lane_center, rotation, WAYPOINT_COLOR)

        left_boundary = waypoint.transform.transform(carla.Location(y=-lane_width/2))
        right_boundary = waypoint.transform.transform(carla.Location(y=lane_width/2))

        self.draw_box(left_boundary, rotation, BOUNDARY_COLOR)
        self.draw_arrow(lane_center, left_boundary, BOUNDARY_COLOR)
        self.draw_arrow(lane_center, right_boundary, BOUNDARY_COLOR)

        bounding_box = av.bounding_box
        av_transform = av.get_transform()
        top_left_corner = av_transform.transform(carla.Location(x=bounding_box.extent.x, y=-bounding_box.extent.y))
        top_right_corner = av_transform.transform(carla.Location(x=bounding_box.extent.x, y=bounding_box.extent.y))
        bottom_left_corner = av_transform.transform(carla.Location(x=-bounding_box.extent.x, y=-bounding_box.extent.y))
        bottom_right_corner = av_transform.transform(carla.Location(x=-bounding_box.extent.x, y=bounding_box.extent.y))

        corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

        self.draw_box(top_left_corner, rotation, CORNER_COLOR)

        AXIS_LENGTH = 0.2
        top_left_corner_transform = carla.Transform(top_left_corner, rotation)
        x_axis = top_left_corner_transform.transform(loc(1, 0, 0)*AXIS_LENGTH)
        y_axis = top_left_corner_transform.transform(loc(0, 1, 0)*AXIS_LENGTH)

        self.draw_arrow(top_left_corner, x_axis, AXIS_X_COLOR, arrow_size = 0.02)
        self.draw_arrow(top_left_corner, y_axis, AXIS_Y_COLOR, arrow_size = 0.02)

        # I will work with rotation matrices, but it would be so cool to work with quaternions instead.
        top_left_to_left_bnd_dist = -compute_lateral_distance(top_left_corner, left_boundary, waypoint.transform)
        print("top_left_to_left_bnd_dist: ", top_left_to_left_bnd_dist)
        
        min_dist_left_bnd = self.min_distance_to_boundary(corners, left_boundary, waypoint.transform, True)
        min_dist_right_bnd = self.min_distance_to_boundary(corners, right_boundary, waypoint.transform, False)

        print("min_dist_left_bnd: ", min_dist_left_bnd)
        print("min_dist_right_bnd: ", min_dist_right_bnd)

        #print("vehicle center location: ", location)
        #print("lane_center: ", lane_center)
        #print("av_transform.location: ", av_transform.location)
        #print("top_left_corner: ", top_left_corner)
        #print("left_boundary: ", left_boundary)
        #bottom_left_to_left_bnd_dist = self.get_distance_to_left_boundary(bottom_left_corner, waypoint.transform.rotation, left_boundary)
        #print("top_left_to_left_bnd_dist: ", top_left_to_left_bnd_dist)
        # print("bottom_left_to_left_bnd_dist: ", bottom_left_to_left_bnd_dist)

        # top_right_to_right_bnd_dist = self.get_distance_to_right_boundary(top_right_corner, waypoint.transform.rotation, right_boundary)
        # bottom_right_to_right_bnd_dist = self.get_distance_to_right_boundary(bottom_right_corner, waypoint.transform.rotation, right_boundary)
        # print("top_right_to_right_bnd_dist: ", top_right_to_right_bnd_dist)
        # print("bottom_right_to_right_bnd_dist: ", bottom_right_to_right_bnd_dist)

        # Let's understand how translation works!
        # Just add some test cases.
        # Next steps: let's compute distance from AV's bounding box to the left and right boundary.
        # And let's test that, ensuring that when I get close or even breach the boundary, the distance is correct.
        # Then similarly compute the distance from AV's center to lane center and these will be my features.
        # Then collect driving data with these values and train the model.
        # Model's output is steering and acceleration.
        # However I need to drive straight right. Perhaps I need information about such distance within some horizon.
        # Also, let's extract out this new logic into a separate file.

        # TODO Jun 23:
        # 1. Debug transformation. Perhaps just write unit tests. Can try using quaternions.
        # 2. Once that is complete. Fun sub-project is to implement camera controls.
        # 3. Then I need to compute features for ML. Boundary distances are good. Need to know future curvatures, need to know current speed, current heading delta.
        

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        blueprint = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2017") #random.choice(blueprint_list)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            #spawn_points = self.map.get_spawn_points()
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = carla.Transform(carla.Location(6.0, 306.6, 0.5), carla.Rotation(0.0, 1.3788, 0.0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()