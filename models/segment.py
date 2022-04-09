
class segment():
    def __init__(self, start, end, axis, filename): 
        
        self.start = start #Sample at which the detected event starts.
        self.end = end #Sample at which the detected event finishes.
        self.axis = axis #Axis where the event was detected.
        self.filename = filename #Filename where the event was detected.
        
        self.data = None
        self.id = None
        self.ax = None
        self.ay = None
        self.az = None
        self.pressure = None
        self.times = None
        
        self.upper_threshold_ax = None
        self.upper_threshold_ay = None
        self.upper_threshold_az = None
        self.lower_threshold_ax = None
        self.lower_threshold_ay = None
        self.lower_threshold_az = None
        
        self.min_ax = None
        self.max_ax = None
        self.min_ay = None
        self.max_ay = None
        self.min_az = None
        self.max_az = None
        self.min_pressure = None
        self.max_pressure = None
        
        self.group_label = None
        
        self.latitude = None
        self.longitude = None
        
        self.timestamp = None
        
    def setup_acceleration(self, data):
        self.data = data
        self.ax = self.data.ax[int(float(self.start)):int(float(self.end))]
        self.ay = self.data.ay[int(float(self.start)):int(float(self.end))]
        self.az = self.data.az[int(float(self.start)):int(float(self.end))]
        self.pressure = self.data.pressure[int(float(self.start)):int(float(self.end))]
        
    def setup_thresholds(self, upper_threshold_ax, lower_threshold_ax, upper_threshold_ay, lower_threshold_ay, upper_threshold_az, lower_threshold_az):
        self.upper_threshold_ax = upper_threshold_ax
        self.upper_threshold_ay = upper_threshold_ay
        self.upper_threshold_az = upper_threshold_az
        
        self.lower_threshold_ax = lower_threshold_ax
        self.lower_threshold_ay = lower_threshold_ay
        self.lower_threshold_az = lower_threshold_az
        
    def setup_gps_data(self, data):
        self.latitude = self.data.latitude[int(float(self.start)):int(float(self.end))]
        self.longitude = self.data.longitude[int(float(self.start)):int(float(self.end))]
        
    def setup_timestamp(self, data):
        self.timestamp = self.data.timestamp[int(float(self.start)):int(float(self.end))]