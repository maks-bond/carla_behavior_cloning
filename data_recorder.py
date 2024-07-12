import datetime
import pandas as pd

class DataEntry:
    __slots__ = [
        'min_dist_to_left_bnd', 'min_dist_to_right_bnd', 'heading_delta', 'speed',
        'curvature_5', 'curvature_10', 'curvature_15', 'curvature_20',
        'curvature_25', 'accel', 'steering_angle'
    ]

class DataRecorder:
    def __init__(self):
        self.data = {}
        self.paused = True

    def start_recording(self):
        self.paused = False

    def pause_recording(self):
        self.paused = True

    def toggle_recording(self):
        if self.paused:
            self.start_recording()
        else:
            self.pause_recording()

    def record_features(self, timestamp, features):
        if self.paused:
            return

        if timestamp not in self.data:
            self.data[timestamp] = DataEntry()

        self.data[timestamp].min_dist_to_left_bnd = features.min_dist_left_bnd
        self.data[timestamp].min_dist_to_right_bnd = features.min_dist_right_bnd
        self.data[timestamp].heading_delta = features.heading_delta
        self.data[timestamp].speed = features.speed
        self.data[timestamp].curvature_5 = features.curvature_5
        self.data[timestamp].curvature_10 = features.curvature_10
        self.data[timestamp].curvature_15 = features.curvature_15
        self.data[timestamp].curvature_20 = features.curvature_20
        self.data[timestamp].curvature_25 = features.curvature_25

    def record_control(self, timestamp, accel, steering_angle):
        if self.paused:
            return

        if timestamp not in self.data:
            self.data[timestamp] = DataEntry()

        self.data[timestamp].accel = accel
        self.data[timestamp].steering_angle = steering_angle

    def write_to_file(self):
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([
            {
                'timestamp': timestamp,
                'min_dist_to_left_bnd': entry.min_dist_to_left_bnd,
                'min_dist_to_right_bnd': entry.min_dist_to_right_bnd,
                'heading_delta': entry.heading_delta,
                'speed': entry.speed,
                'curvature_5': entry.curvature_5,
                'curvature_10': entry.curvature_10,
                'curvature_15': entry.curvature_15,
                'curvature_20': entry.curvature_20,
                'curvature_25': entry.curvature_25,
                'accel': entry.accel,
                'steering_angle': entry.steering_angle
            }
            for timestamp, entry in self.data.items()
        ])

        # Sort the DataFrame by timestamp
        df = df.sort_values(by='timestamp')

        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

        file_path = f'data_{timestamp}.csv'

        df.to_csv(file_path, index=False)

        print(f'DataFrame has been written to {file_path}')