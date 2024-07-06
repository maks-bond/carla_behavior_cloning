import re
import carla
import random
import weakref
import math
import collections

import numpy as np

def get_lane_ids_to_next_goal():
    