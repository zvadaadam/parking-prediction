import numpy as np
import pandas as pd
import holidays
import requests
from tqdm import tqdm
from datetime import datetime
from parkingprediction.dataset.base_dataset import DatasetBase


class ParkingDataset(DatasetBase):

    def __init__(self, config):
        super(ParkingDataset, self).__init__(config)

        self.load_dataset()

    def load_dataset(self):
        self.df = self.create_dataset()
        self.split_dataset()

    def create_dataset(self, csv_path=None):

        if csv_path == None:
            csv_path = self.config.csv_path()

        print(csv_path)

        df = pd.read_csv(csv_path)

        df = df.drop(columns='value1')
        df = df.set_index('created')
        df = df.drop(columns='id')
        df.index = pd.to_datetime(df.index)
        df = df.groupby(pd.Grouper(freq='h')).mean()

        df = self.update_external_features(df)

        return df

    def update_external_features(self, df):

        weather_features = self.get_weather()

        days_features = self.get_days_of_week()

        temp_feature = [weather[0] for weather in weather_features]
        df['temperature'] = temp_feature

        pressure_feature = [weather[1] for weather in weather_features]
        df['pressure'] = pressure_feature

        mo = []
        tue = []
        wen = []
        thr = []
        fr = []
        sat = []
        sun = []
        holiday = []

        for days in days_features:
            mo.append(days[0])
            tue.append(days[1])
            wen.append(days[2])
            thr.append(days[3])
            fr.append(days[4])
            sat.append(days[5])
            sun.append(days[6])
            holiday.append(days[7])

        df['monday'] = mo
        df['tuesday'] = tue
        df['wednesday'] = wen
        df['thursday'] = thr
        df['friday'] = fr
        df['saturday'] = sat
        df['sunday'] = sun
        df['holiday'] = holiday

        return df

    def get_weather(self, start="2018-01-01", end="2018-12-31"):
        datelist = pd.date_range(start=start, end=end).tolist()
        return_dict = {}
        for i in datelist:
            START_DATE = i.strftime('%Y%m%d')
            END_DATE = i.strftime('%Y%m%d')
            URL = "https://api.weather.com/v1/geocode/50.12/14.54/observations/historical.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&startDate={}&endDate={}&units=m".format(
                START_DATE, END_DATE)
            r = requests.get(URL)
            observations = r.json()['observations']
            for i in observations:
                dt_object = datetime.fromtimestamp(i['valid_time_gmt'])
                t = (i['temp'], i['pressure'])
                return_dict[str(dt_object)] = t

        datelist = pd.date_range(start="2018-01-01", end="2018-12-31").tolist()
        timelist = []
        for i in datelist:
            for h in range(0, 24):
                datestr = i.strftime('%Y-%m-%d')
                timelist.append("{} {:02d}:00:00".format(datestr, h))

        return_list = []
        prev_val = None
        for t in timelist:
            if t in return_dict:
                return_list.append(return_dict[t])
                prev_val = return_dict[t]
            else:
                return_list.append(prev_val)
        return return_list

    def get_days_of_week(self, start="2018-01-01", end="2018-12-31"):
        cz_holidays = holidays.CZ()

        date_list = pd.date_range(start=start, end=end).tolist()
        output_list = []
        for d in date_list:
            weekday = int(d.strftime('%w'))
            for _ in range(0, 24):
                output_list.append(
                    [int(weekday == 1), int(weekday == 2), int(weekday == 3), int(weekday == 4), int(weekday == 5),
                     int(weekday == 6), int(weekday == 0), int(d in cz_holidays)])
        return output_list


    def normalization(self, coef):

        coef_norm = coef - coef.mean()
        coef_norm = coef_norm / coef_norm.max()

        return coef_norm


if __name__ == '__main__':

    from parkingprediction.config.config_reader import ConfigReader

    dataset = ParkingDataset(ConfigReader())

    print(dataset.train_df)