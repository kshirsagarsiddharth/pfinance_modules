import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cufflinks as cf 
import numba as nb 
import plotly.offline
from plotly import tools 
import ipywidgets as widgets 
from IPython.display import display
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import datetime

class ITStocks:
    def __init__(self) -> None:
    
        self.df = pd.read_csv("../IT.NS.CSV", parse_dates=True)
        self.COMPANY_LIST = ['MPHASIS.NS', 'COFORGE.NS', 'MINDTREE.NS', 'INFY.NS', 'TECHM.NS',
        'LTI.NS', 'HCLTECH.NS', 'TCS.NS', 'WIPRO.NS', 'LTTS.NS']


    def select_time_period(self,df,code):
        """
        Resamples the dataframe for given code
        :param df: dataframe to resample
        :param code: time period to resample for
        """
        df = df.resample(rule = code).last()
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
        
    def pick_company_candles(self,company_name, time_period_start,time_period_end, time_offset):
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
        qf = cf.QuantFig(df2[['Open','High','Low','Close']])
        
        #qf.add_bollinger_bands()
        qf.add_rsi(periods=14, yTitle='RSI')
        qf.iplot(theme='solar', up_color='green', down_color='red', dimensions = (1500,600),
        width = 6,title = company_name

        )

    def display_vizs(self):
        controls = widgets.interact(self.pick_company,
                                company_name = widgets.Dropdown(options = self.COMPANY_LIST,index = 2),
                                time_period_start = widgets.DatePicker(description = 'Pick a time'),
                                time_period_end = widgets.DatePicker(description = 'Pick a time'),
                                time_offset = widgets.ToggleButtons(
                                                                    options=['D','W','M','Q','A'],
                                                                    description='Time Offset:',
                                                                    disabled=False,
                                                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                                                    tooltips=['Daily', 'Weekly', 'Monthly','Annually'],
                                                                #     icons=['check'] * 3
                                                                )
                            )
    def display_vizs_candles(self):
        controls = widgets.interact(self.pick_company_candles,
                                company_name = widgets.Dropdown(options = self.COMPANY_LIST,index = 2),
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

class AutoStocks(ITStocks):
    def __init__(self) -> None:
        self.df = pd.read_csv("../AUTO.NS.CSV", parse_dates=True) 
        self.COMPANY_LIST = ['TVSMOTOR.NS', 'TATAMOTORS.NS',
       'BAJAJ-AUTO.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'MARUTI.NS',
       'EICHERMOT.NS', 'BOSCHLTD.NS', 'BALKRISIND.NS', 'ASHOKLEY.NS',
       'EXIDEIND.NS', 'BHARATFORG.NS', 'AMARAJABAT.NS', 'MRF.NS',
       'TIINDIA.NS']


class BankStocks(ITStocks): 
    def __init__(self) -> None:
        self.df = pd.read_csv("../BANK.NS.CSV", parse_dates=True)
        self.COMPANY_LIST = ['BANDHANBNK.NS',
       'KOTAKBANK.NS', 'HDFCBANK.NS', 'SBIN.NS', 'FEDERALBNK.NS',
       'RBLBANK.NS', 'IDFCFIRSTB.NS', 'INDUSINDBK.NS', 'AXISBANK.NS',
       'PNB.NS', 'AUBANK.NS', 'ICICIBANK.NS']

class PharmaStocks(ITStocks): 
    def __init__(self) -> None:
        self.df = pd.read_csv("../PHARMA.NS.CSV", parse_dates=True)
        self.COMPANY_LIST = ['ABBOTINDIA.NS', 'APLLTD.NS', 'ALKEM.NS', 'AUROPHARMA.NS',
       'BIOCON.NS', 'CADILAHC.NS', 'CIPLA.NS', 'DIVISLAB.NS',
       'DRREDDY.NS', 'GLAND.NS', 'GLENMARK.NS', 'GRANULES.NS',
       'IPCALAB.NS', 'LAURUSLABS.NS', 'LUPIN.NS', 'NATCOPHARM.NS',
       'PFIZER.NS', 'STAR.NS', 'SUNPHARMA.NS', 'TORNTPHARM.NS']

class NiftyFifty(ITStocks):
    def __init__(self) -> None:
        self.df = pd.read_csv("../NIFTY50.CAP.NS.CSV", parse_dates=True)
        self.COMPANY_LIST = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
       'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
       'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS',
       'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS',
       'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
       'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'IOC.NS',
       'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
       'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS',
       'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS',
       'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS',
       'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS',
       'ULTRACEMCO.NS', 'WIPRO.NS']




class DisplayReturns():
    """
    Displays Returns of all sectors
    """
    def __init__(self) -> None:
        self.display_returns()
    
    def equally_weighted_returns(self,stock_object):
        df = stock_object.df
        df = df.set_index('Date')
        all_close_values = []
        N = len(stock_object.COMPANY_LIST)
        for i in range(N):
            temp_df = df[df['ticker'] == stock_object.COMPANY_LIST[i]]['Adj Close'].pct_change() 
            #temp_df = (temp_df.cumprod() + 1) - 1
            #print(temp_df.isnull().sum(), stock_object.COMPANY_LIST[i],stock_object.__class__.__name__)
            all_close_values.append(temp_df)
        #print("\n")
        combined = pd.concat(all_close_values, axis = 1)
        combined = combined.fillna(0)
        #combined = (combined + 1).cumprod() - 1
        equal_weights = np.repeat(1/N,N)
        equal_returns = combined.mul(equal_weights, axis = 1).sum(axis = 1)
        equal_returns = equal_returns.to_frame(stock_object.__class__.__name__)
        return equal_returns 

    def compare_equally_weighted_returns(self,object_iterator): 
        df = []
        for stock_object in object_iterator: 
            ewr = self.equally_weighted_returns(stock_object)
            df.append(ewr)
        df2 = pd.concat(df, axis = 1)
        return df2.dropna()
    def display_returns(self):
        objects = (ITStocks(), AutoStocks(), BankStocks(), PharmaStocks(), NiftyFifty())
        equal_returns = self.compare_equally_weighted_returns(objects)
        equal_cumulative_returns = (equal_returns + 1).cumprod() - 1
        equal_returns.iplot(kind = 'histogram', theme = 'solar')
        equal_cumulative_returns.iplot(kind = 'histogram', theme = 'solar')
        equal_cumulative_returns.iplot(theme = 'solar')