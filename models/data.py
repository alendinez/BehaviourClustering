from scipy import signal

### Data model for acceleration data.
class data():
    def __init__(self, filename, ax, ay, az):
        
        self.filename = filename
        self.ax = ax
        self.ay = ay
        self.az = az
        
        self.pressure = None
        self.timestamp = None
        self.latitude = None
        self.longitude = None
        
    ### Filter acceleration signals using a Butterworth filter.   
    def filter_accelerations(self, N, Wn):
        b, a = signal.butter(N, Wn)
        
        self.ax = signal.filtfilt(b, a, self.ax)
        self.ay = signal.filtfilt(b, a, self.ay)
        self.az = signal.filtfilt(b, a, self.az)
        
        
        
