import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cufflinks as cf 
import numba as nb 
import plotly.offline
import ipywidgets as widgets 
from IPython.display import display
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import datetime

class ITStocks:
    def __init__(self) -> None:
    
        self.df = pd.read_csv("../IT.NS.CSV")
        self.IT_COMPANY_LIST = ['MPHASIS.NS', 'COFORGE.NS', 'MINDTREE.NS', 'INFY.NS', 'TECHM.NS',
        'LTI.NS', 'HCLTECH.NS', 'TCS.NS', 'WIPRO.NS', 'LTTS.NS']


    def select_time_period(self,df,code):
        """
        Resamples the dataframe for given code
        :param df: dataframe to resample
        :param code: time period to resample for
        """
        df = df.resample(rule = code).mean()
        return df

    def pick_company(self,company_name, time_period_start,time_period_end, time_offset):
        """
        :param company_name: company to pick 
        :param time_period: time period used for visualization
        """
        df2 = self.df[self.df['ticker'] == company_name]
        df2['Date'] = pd.to_datetime(df2['Date'])
    
        df2 = df2.set_index('Date')
        df2 = self.select_time_period(df2, time_offset)
        
            
        if time_period_start or time_period_end:
            df2 = df2.loc[time_period_start:time_period_end]
            
        df2 = df2.dropna()
        df2['Adj Close'] = (df2['Adj Close']) #/ df2['Adj Close'].iloc[0]) * 100
        df2['cum_returns'] = df2['Adj Close'].pct_change()
        df2[['Adj Close','Volume','cum_returns']].iplot(theme = 'solar',
                                    dimensions = (1000,600),
                                    subplots = True,
                                    xTitle = 'Date'
                        )

    def display_vizs(self):
        controls = widgets.interact(self.pick_company,
                                company_name = widgets.Dropdown(options = self.IT_COMPANY_LIST,index = 2),
                                time_period_start = widgets.DatePicker(description = 'Pick a time'),
                                time_period_end = widgets.DatePicker(description = 'Pick a time'),
                                time_offset = widgets.ToggleButtons(
                                                                    options=['D','W','M','Q','A'],
                                                                    description='Time Offset:',
                                                                    disabled=False,
                                                                    button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
                                                                    tooltips=['Daily', 'Weekly', 'Monthly','Annually'],
                                                                #     icons=['check'] * 3
                                                                )
                            )

    