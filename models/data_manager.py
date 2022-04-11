import csv
import datetime
import numpy as np
import pandas as pd
from scipy import signal

import models.segment as sgmnt
import models.data as dt


class data_manager():
    def __init__(self):
        pass
    
    ### Load acceleration data from .csv file
    def load_data(self, filename, path):
        try:
            data_pd = pd.read_csv(path + filename + '.csv', header=0, sep='\t')
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
        except:
            data_pd = pd.read_csv(path + filename + '.csv', header=0)
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
            
        data = dt(filename, accelerations[:,0], accelerations[:,1], accelerations[:,2])
        data.pressure = np.array(data_pd.loc[:, ['Pressure']].values)
        return data
    
    ### Load acceleration data including pressure, latitude, longitude and timestamp
    def load_data_gps(self, filename, path):
        try:
            data_pd = pd.read_csv(path + filename + '.csv', header=0, sep=';')
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
        except:
            data_pd = pd.read_csv(path + filename + '.csv', header=0)
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
            
        data = dt(filename, accelerations[:,0], accelerations[:,1], accelerations[:,2])
        
        data.pressure = np.array(data_pd.loc[:, ['Pressure']].values)
        
        latitude = data_pd.loc[:, ['location-lat']].values
        longitude = data_pd.loc[:, ['location-lon']].values
        data.latitude = np.array(latitude)
        data.longitude = np.array(longitude)
        '''
        try:
            datetime_format = "%d/%m/%Y %H:%M:%S.%f"
            temp_datetimes = data_pd.loc[:, ['Timestamp']].values
            datetimes = []
            for dtime in temp_datetimes:
                dtime = datetime.datetime.strptime(dtime[0], datetime_format)
                datetimes.append(dtime)
        except:
            datetime_format = "%d-%m-%Y %H:%M:%S.%f"
            temp_datetimes = data_pd.loc[:, ['Timestamp']].values
            datetimes = []
            for dtime in temp_datetimes:
                dtime = datetime.datetime.strptime(dtime[0], datetime_format)
                datetimes.append(dtime)
            
        data.timestamp = datetimes
        '''   
        return data
    
    ### Load acceleration data including pressure, latitude, longitude and timestamp
    def load_data_gps_timestamp(self, filename, path):
        try:
            data_pd = pd.read_csv(path + filename + '.csv', header=0, sep=';')
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
        except:
            data_pd = pd.read_csv(path + filename + '.csv', header=0)
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
            
        data = dt(filename, accelerations[:,0], accelerations[:,1], accelerations[:,2])
        
        data.pressure = np.array(data_pd.loc[:, ['Pressure']].values)
        
        latitude = data_pd.loc[:, ['location-lat']].values
        longitude = data_pd.loc[:, ['location-lon']].values
        data.latitude = np.array(latitude)
        data.longitude = np.array(longitude)
        
        try:
            datetime_format = "%d/%m/%Y %H:%M:%S.%f"
            temp_datetimes = data_pd.loc[:, ['Timestamp']].values
            datetimes = []
            for dtime in temp_datetimes:
                dtime = datetime.datetime.strptime(dtime[0], datetime_format)
                datetimes.append(dtime)
        except:
            datetime_format = "%d-%m-%Y %H:%M:%S.%f"
            temp_datetimes = data_pd.loc[:, ['Timestamp']].values
            datetimes = []
            for dtime in temp_datetimes:
                dtime = datetime.datetime.strptime(dtime[0], datetime_format)
                datetimes.append(dtime)
            
        data.timestamp = datetimes
           
        return data
    
    def load_data_datetimes(self, filename, path):
        try:
            data_pd = pd.read_csv(path + filename + '.csv', header=0, sep=';')
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
        except:
            data_pd = pd.read_csv(path + filename + '.csv', header=0)
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
            
        data = dt(filename, accelerations[:,0], accelerations[:,1], accelerations[:,2])
        
        data.pressure = np.array(data_pd.loc[:, ['Pressure']].values)
        
        try:
            datetime_format = "%d/%m/%Y %H:%M:%S.%f"
            temp_datetimes = data_pd.loc[:, ['Timestamp']].values
            datetimes = []
            for dtime in temp_datetimes:
                dtime = datetime.datetime.strptime(dtime[0], datetime_format)
                datetimes.append(dtime)
        except:
            datetime_format = "%d-%m-%Y %H:%M:%S.%f"
            temp_datetimes = data_pd.loc[:, ['Timestamp']].values
            datetimes = []
            for dtime in temp_datetimes:
                dtime = datetime.datetime.strptime(dtime[0], datetime_format)
                datetimes.append(dtime)
            
        data.timestamp = datetimes
         
        return data
        
        
    def load_segments(self, filename, path, sigma, w):
        csv_filename = "events_sigma"+str(sigma)+"_w"+str(w)+"_"+filename+".csv" 
        pathfile = path + csv_filename
        
        segments = []
        with open(pathfile, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != None:
                i = 0
                for row in reader:
                    if i % 2 != 0:
                        segments.append(row)
                    i = i + 1
            
        print("Total number of segments: "+str(len(segments))) 
        return segments
    
    def load_all_segments(self, path, sigma, w):
        csv_filename = "allsegments_sigma"+str(sigma)+"_w"+str(w)+".csv" 
        pathfile = path + csv_filename
        
        segments = []
        with open(pathfile, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != None:
                i = 0
                for row in reader:
                    if i % 2 != 0:
                        segments.append(row)
                    i = i + 1
                    
        all_segments = []
        for segment in segments:
            current_segment = sgmnt(int(float(segment[1])), int(float(segment[2])), segment[3], segment[4])
            current_segment.id = int(segment[0])
            all_segments.append(current_segment)
        
        print("Total number of segments loaded: "+str(len(all_segments))) 
        return all_segments

    def load_all_segments_linux(self, path, sigma, w):
        csv_filename = "allsegments_sigma"+str(sigma)+"_w"+str(w)+".csv" 
        pathfile = path + csv_filename
        
        segments = []
        with open(pathfile, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != None:
                i = 0
                for row in reader:
                    segments.append(row)
                    i = i + 1
                    
        all_segments = []
        for segment in segments:
            current_segment = sgmnt(int(float(segment[1])), int(float(segment[2])), segment[3], segment[4])
            current_segment.id = int(segment[0])
            all_segments.append(current_segment)
        
        print("Total number of segments loaded: "+str(len(all_segments))) 
        return all_segments
    
    def load_corr_data(self, path, sigma, w, number_of_segments):
        import csv
        import numpy as np
        
        '''
        corr_ax_filename = 'corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        corr_ay_filename = 'corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        corr_az_filename = 'corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        '''
        corr_ax_filename = 'corr_ax.csv'
        corr_ay_filename = 'corr_ay.csv'
        corr_az_filename = 'corr_az.csv'
        lag_ax_filename = 'lag_ax.csv'
        
        #lag_ax_filename = 'lag_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        
        corr_ax_pathfile = path + corr_ax_filename
        corr_ay_pathfile = path + corr_ay_filename
        corr_az_pathfile = path + corr_az_filename
        lag_ax_pathfile = path + lag_ax_filename
        
        corr_ax, corr_ay, corr_az, lag_ax = [], [], [], []
        
        corr_ax_csv = open(corr_ax_pathfile)
        corr_ax_reader = csv.reader(corr_ax_csv)
        corr_ax = list(corr_ax_reader)[0]
        length = len(corr_ax)
        corr_ax = np.array([corr_ax[x:x+number_of_segments] for x in range(0, length, number_of_segments)])
                    
        corr_ay_csv = open(corr_ay_pathfile)
        corr_ay_reader = csv.reader(corr_ay_csv)
        corr_ay = list(corr_ay_reader)[0]
        corr_ay = np.array([corr_ay[x:x+number_of_segments] for x in range(0, length, number_of_segments)])
                    
        corr_az_csv = open(corr_az_pathfile)
        corr_az_reader = csv.reader(corr_az_csv)
        corr_az = list(corr_az_reader)[0]
        corr_az = np.array([corr_az[x:x+number_of_segments] for x in range(0, length, number_of_segments)])
                    
        lag_ax_csv = open(lag_ax_pathfile)
        lag_ax_reader = csv.reader(lag_ax_csv)
        lag_ax = list(lag_ax_reader)[0]
        lag_ax = np.array([lag_ax[x:x+number_of_segments] for x in range(0, length, number_of_segments)])
                    
        return corr_ax, corr_ay, corr_az, lag_ax
    
    ### Method to export the events to .csv
    def export_segments(self, segments, sigma, w, filename, path):
        fields = ['id', 'start', 'end', 'axis', 'filename']
        export_filename = path+"segments_sigma"+str(sigma)+"_w"+str(w)+"_"+filename+".csv"
        
        with open(export_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for segment in segments:
                writer.writerow([segment.id, segment.start, segment.end, segment.axis, segment.filename])
                
    ### Method to export all the events to .csv
    def export_all_segments(self, segments, sigma, w, path):
        fields = ['id', 'start', 'end', 'axis', 'filename']
        export_filename = path+"allsegments_sigma"+str(sigma)+"_w"+str(w)+".csv"
        
        with open(export_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for segment in segments:
                writer.writerow([segment.id, segment.start, segment.end, segment.axis, segment.filename])
                
                
    def interpolate_gps_data(self, data):
        longitude, latitude = data.longitude, data.latitude
        
        length = len(data.latitude)
        result_longitude, result_latitude = np.zeros(length), np.zeros(length)        
        first_nonzero_lon, first_nonzero_lat = int((longitude!=0).argmax(axis=0)), int((latitude!=0).argmax(axis=0))
        longitude, latitude = longitude[first_nonzero_lon::25], latitude[first_nonzero_lat::25]
        
        t = np.arange(0, length, 25)
        longitude = signal.resample(longitude, length-first_nonzero_lon, t)
        latitude = signal.resample(latitude, length-first_nonzero_lat, t)
        
        result_longitude[first_nonzero_lon:len(result_longitude)] = longitude[0].ravel()
        result_latitude[first_nonzero_lat:len(result_latitude)] = latitude[0].ravel()
        
        return result_longitude, result_latitude
