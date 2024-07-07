import re
import carla
import random
import weakref
import math
import collections

import numpy as np

from data_recorder import DataRecorder

VIZ_Z = 0.3
FIXED_DELTA_SECONDS = 0.05

def draw_box(debug, location, rotation, color):
    BOX_WIDTH = 0.05
    THICKNESS = 0.01
    debug_location = carla.Location(location.x, location.y, location.z+VIZ_Z)
    box = carla.BoundingBox(debug_location, carla.Vector3D(BOX_WIDTH, BOX_WIDTH, BOX_WIDTH))
    debug.draw_box(box, rotation, thickness=THICKNESS, life_time=FIXED_DELTA_SECONDS+0.01, color=color)

def draw_arrow(debug, start_location, end_location, color, arrow_size=0.1):
    THICKNESS = 0.01
    debug_start_location = carla.Location(start_location.x, start_location.y, start_location.z+VIZ_Z)
    debug_end_location = carla.Location(end_location.x, end_location.y, end_location.z+VIZ_Z)
    debug.draw_arrow(debug_start_location, debug_end_location, thickness=THICKNESS, life_time=FIXED_DELTA_SECONDS+0.01, color=color, arrow_size=arrow_size)

def draw_string(debug, location, text, color):
    debug_location = carla.Location(location.x, location.y, location.z+VIZ_Z+0.02)
    debug.draw_string(debug_location, text, life_time=FIXED_DELTA_SECONDS+0.01, color=color)

def get_distance_to_left_boundary(left_corner, rotation, left_boundary):
    transform = carla.Transform(left_corner, rotation)
    bnd_in_corner = transform.transform(carla.Location(x=left_boundary.x, y=left_boundary.y, z=left_boundary.z))

    return -bnd_in_corner.y

def get_distance_to_right_boundary(right_corner, rotation, right_boundary):
    transform = carla.Transform(right_corner, rotation)
    bnd_in_corner = transform.transform(carla.Location(x=right_boundary.x, y=right_boundary.y, z=right_boundary.z))

    return bnd_in_corner.y

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

def compute_lateral_distance(source_point, target_point, target_transform):
    rotation_matrix = transform_to_rotation_matrix(target_transform)
    target_relative_to_source = to_np(target_point-source_point)
    target_relative_to_source[2] = 0.0

    target_in_source = np.dot(rotation_matrix.T, target_relative_to_source)

    return target_in_source[1]

def min_distance_to_boundary(corners, boundary, boundary_transform, is_left):
    min_dist = float("inf")
    for corner in corners:
        lateral_dist = compute_lateral_distance(corner, boundary, boundary_transform)
        if is_left:
            lateral_dist = -lateral_dist
        
        min_dist = min(min_dist, lateral_dist)
    
    return min_dist

# Alternative
# We have global_from_av
# We have global_from_waypoint
# We need waypoint_from_av
# waypoint_from_av = global_from_waypoint.inverse() * global_from_av
def compute_heading_delta(waypoint_transform, av_transform):
    # We will convert av into waypoint frame and then we will get an angle with x axis.
    # However, AV's rotation matters.
    # We can get unit x vector in AV's frame.

    # unit_x = carla.Location(1.0, 0.0, 0.0)
    # unit_x_in_av_frame = av_transform.transform(unit_x)

    # # Now imagine that AV's point has a rotation taken from waypoint transform.
    # # We need to find an angle between unit_x and x axis of waypoint transform placed at AV's point.

    # waypoint_transform_at_av = carla.Transform(av_transform.location, waypoint_transform.rotation)

    # # Global coordinates of unit x vector which is placed 
    # unit_x_in_av_waypoint_frame = waypoint_transform_at_av.transform(unit_x)

    # # a*b = |a|*|b|*cos(theta)
    # unit_x_av_in_av = unit_x_in_av_frame - av_transform.location
    # unit_x_waypoint_in_av = unit_x_in_av_waypoint_frame - av_transform.location

    # unit_x_av_in_av_np = to_np(unit_x_av_in_av)[:2]
    # unit_x_waypoint_in_av_np = to_np(unit_x_waypoint_in_av)[:2]

    # print("unit_x_av_in_av_np is: ", unit_x_av_in_av_np)
    # print("unit_x_waypoint_in_av_np is: ", unit_x_waypoint_in_av_np)

    # theta_cos = np.dot(unit_x_av_in_av_np, unit_x_waypoint_in_av_np)/(np.linalg.norm(unit_x_av_in_av_np)*np.linalg.norm(unit_x_waypoint_in_av_np))

    # theta = math.acos(theta_cos)

    # New approach.
    # wa_from_x = wa_from_global*global_from_av*av_from_x

    global_from_wa = transform_to_rotation_matrix(waypoint_transform)
    global_from_av = transform_to_rotation_matrix(av_transform)
    av_from_x = np.array([1.0,0.0,0.0])

    x_in_wa = np.dot(np.dot(global_from_wa.T, global_from_av),av_from_x)
    
    x_in_wa_2d = x_in_wa[:2]
    unit_x_2d = np.array([1.0, 0.0])
    
    theta_cos = np.dot(x_in_wa_2d, unit_x_2d)/(np.linalg.norm(x_in_wa_2d)*np.linalg.norm(unit_x_2d))
    theta = math.acos(theta_cos)

    sign = 1.0 if x_in_wa[1] >= 0.0 else -1.0
    theta = sign*theta

    return theta

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
    area_sq = s * (s - a) * (s - b) * (s - c)

    # Check if the points are collinear (area == 0)
    if area_sq <= 0.0:
        return 0.0

    area = math.sqrt(area_sq)
    
    # Calculate the circumradius
    circumradius = (a * b * c) / (4 * area)
    
    # Calculate the curvature (1 / circumradius)
    curvature = 1 / circumradius
    
    return curvature

# Perhaps I should build my own route, connecting waypoints.

#road_ids = set()
#road_ids = {0, 1, 2, 3, 390, 12, 13, 14, 15, 16, 17, 18, 19, 276, 411, 32, 47, 62, 351, 252, 383}
road_ids = {32, 1, 0, 3, 2, 12, 13, 14, 15, 16, 17, 18, 19, 276, 411, 383}

def get_waypoint_at_distance_on_route(source_w, distance):
    waypoint = None
    for w in source_w.next(distance):
        if w.road_id in road_ids:
            if waypoint is not None:
                print("Existing waypoint has road id: ", waypoint.road_id)
                print("New waypoint has road id: ", w.road_id)
                raise RuntimeError("multiple waypoints on route")
            waypoint = w
    
    if waypoint is None:
        print("road ids:")
        for w in source_w.next(distance):
            print(w.road_id)
        raise RuntimeError("no waypoints on route")
    
    return waypoint

def waypoint_to_2d_point(waypoint):
    location = waypoint.transform.location
    return (location.x, location.y)

def curvature_at_distance(source_w, distance):
    #return 0.0

    STEP = 0.2
    w1 = get_waypoint_at_distance_on_route(source_w, distance - STEP)
    w2 = get_waypoint_at_distance_on_route(source_w, STEP)
    w3 = get_waypoint_at_distance_on_route(source_w, distance + STEP)

    curvature = calculate_curvature(waypoint_to_2d_point(w1), waypoint_to_2d_point(w2), waypoint_to_2d_point(w3))

    return curvature

def compute_features(player, map, debug, timestamp, data_recorder):
    # next_action = self.traffic_manager.get_next_action(self.player)
    # print("Next action is: ", next_action)

    av = player
    location = player.get_location()
    transform = player.get_transform()
    rotation = transform.rotation

    # print("Location x: ", location.x, ", locataion.y: ", location.y, ", location.z: ", location.z)
    # print("rotation pitch: ", rotation.pitch, ", rotation.yaw: ", rotation.yaw, ", rotation.roll: ", rotation.roll)

    WAYPOINT_COLOR = carla.Color(255,0,0,100)
    CORNER_COLOR = carla.Color(0,0,255,100)
    BOUNDARY_COLOR = carla.Color(0,255,0,100)
    
    AXIS_X_COLOR = carla.Color(255, 5, 5, 100)
    AXIS_Y_COLOR = carla.Color(255, 105, 105, 100)
    AXIS_Z_COLOR = carla.Color(255, 205, 205, 100)

    # Waypoint represents closest lane sample to given location.
    waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

    # print("Road id: ", waypoint.road_id)

    road_ids.add(waypoint.road_id)

    # print("W at 100.0")
    # for next_w in waypoint.next(100.0):
    #     print(next_w.road_id)

    lane_center = waypoint.transform.location
    rotation = waypoint.transform.rotation

    lane_width = waypoint.lane_width

    draw_box(debug, lane_center, rotation, WAYPOINT_COLOR)

    left_boundary = waypoint.transform.transform(carla.Location(y=-lane_width/2))
    right_boundary = waypoint.transform.transform(carla.Location(y=lane_width/2))

    draw_box(debug, left_boundary, rotation, BOUNDARY_COLOR)
    draw_arrow(debug, lane_center, left_boundary, BOUNDARY_COLOR)
    draw_arrow(debug, lane_center, right_boundary, BOUNDARY_COLOR)

    bounding_box = av.bounding_box
    av_transform = av.get_transform()
    top_left_corner = av_transform.transform(carla.Location(x=bounding_box.extent.x, y=-bounding_box.extent.y))
    top_right_corner = av_transform.transform(carla.Location(x=bounding_box.extent.x, y=bounding_box.extent.y))
    bottom_left_corner = av_transform.transform(carla.Location(x=-bounding_box.extent.x, y=-bounding_box.extent.y))
    bottom_right_corner = av_transform.transform(carla.Location(x=-bounding_box.extent.x, y=bounding_box.extent.y))

    corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

    draw_box(debug, top_left_corner, rotation, CORNER_COLOR)

    AXIS_LENGTH = 0.2
    top_left_corner_transform = carla.Transform(top_left_corner, rotation)
    x_axis = top_left_corner_transform.transform(loc(1, 0, 0)*AXIS_LENGTH)
    y_axis = top_left_corner_transform.transform(loc(0, 1, 0)*AXIS_LENGTH)

    draw_arrow(debug, top_left_corner, x_axis, AXIS_X_COLOR, arrow_size = 0.02)
    draw_arrow(debug, top_left_corner, y_axis, AXIS_Y_COLOR, arrow_size = 0.02)

    # I will work with rotation matrices, but it would be so cool to work with quaternions instead.
    top_left_to_left_bnd_dist = -compute_lateral_distance(top_left_corner, left_boundary, waypoint.transform)
    #print("top_left_to_left_bnd_dist: ", top_left_to_left_bnd_dist)
    
    min_dist_left_bnd = min_distance_to_boundary(corners, left_boundary, waypoint.transform, True)
    min_dist_right_bnd = min_distance_to_boundary(corners, right_boundary, waypoint.transform, False)

    #print("min_dist_left_bnd: ", min_dist_left_bnd)
    #print("min_dist_right_bnd: ", min_dist_right_bnd)

    heading_delta = compute_heading_delta(waypoint.transform, transform)

    speed = 0.0

    curvature_5 = curvature_at_distance(waypoint, 5.0)
    curvature_10 = curvature_at_distance(waypoint, 10.0)
    curvature_15 = curvature_at_distance(waypoint, 15.0)
    curvature_20 = curvature_at_distance(waypoint, 20.0)
    curvature_25 = curvature_at_distance(waypoint, 25.0)

    print("curvature_5: ", curvature_5)
    print("curvature_10: ", curvature_10)
    print("curvature_15: ", curvature_15)
    print("curvature_20: ", curvature_20)
    print("curvature_25: ", curvature_25)


    data_recorder.record_features(timestamp, min_dist_left_bnd, min_dist_right_bnd, heading_delta, speed, curvature_5, curvature_10, curvature_15, curvature_20, curvature_25)

    # 2. Once that is complete. Fun sub-project is to implement camera controls.
    # 3. Then I need to compute features for ML. Boundary distances are good. Need to know future curvatures, need to know current speed, current heading delta.