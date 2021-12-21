import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cufflinks as cf 
import numba as nb 
import plotly.offline
from plotly import tools 
import ipywidgets as widgets 
from IPython.display import display
from . import pfinance_risk_kit as pfk
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima

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


class DisplayRiskRatios: 
    a = widgets.Checkbox(
    value=True,
    description='drawdown',
    disabled=False
    )

    b = widgets.Checkbox(
        value=True,
        description='skewness',
        disabled=False
    )

    c = widgets.Checkbox(
        value=False,
        description='kurtosis',
        disabled=False
    )
    d = widgets.Checkbox(
        value=False,
        description='value_at_risk',
        disabled=False
    )

    e = widgets.Checkbox(
        value=False,
        description='annualized_sharp_ratio',
        disabled=False
    )

    f = widgets.Checkbox(
        value=False,
        description='annualized_returns',
        disabled=False
    )
    g = widgets.Checkbox(
        value=False,
        description='annualized_volatility',
        disabled=False
    )

    def __init__(self, sector_object) -> None:
        self.df = sector_object.df 
        self.COMPANY_LIST = sector_object.COMPANY_LIST 
        self.RISK_DATAFRAME = self.company_risk_dataframe().set_index('company_name') 
        self.RISK_MEASURE = ['drawdown',
                            'skewness',
                            'kurtosis',
                            'value_at_risk',
                            'annualized_returns',
                            'annualize_volatility',
                            'annualized_sharp_ration']
        self.display_risk_measures()

    def return_risk_summary(self,company_name,r): 
        """
        Compute all risk ratios
        """
        drawdown = pfk.drawdown(r)['drawdown'].min() 
        skewness = pfk.skewness(r)
        kurtosis = pfk.semi_deviation(r) 
        var = pfk.var_parametric_cornsih_fisher(r) 
        annualized_returns = pfk.annualize_returns(r,periods_per_year=365) 
        annualized_volatility = pfk.annualize_volatility(r, periods_per_year=365) 
        annualized_sharp_ratio = pfk.sharpe_ratio(r, periods_per_year=365, riskfree_rate = 0.03) 
        return {   
            'company_name': company_name,
            'drawdown' : drawdown,
            'skewness' : skewness,
            'kurtosis': kurtosis,
            'value_at_risk': var, 
            'annualized_returns': annualized_returns, 
            'annualize_volatility': annualized_volatility,
            'annualized_sharp_ration': annualized_sharp_ratio         
        }
    
    def return_risk(self,company_name):
        """
        :param company_name: company to pick 
        returns the risk associated with each company
        """

        df2 = self.df[self.df['ticker'] == company_name]
        df2['Date'] = pd.to_datetime(df2['Date'])
    
        df2 = df2.set_index('Date')
        df2['returns'] = df2['Adj Close'].pct_change()
        df2 = df2.dropna()
        return self.return_risk_summary(company_name, df2['returns'])
    
    def company_risk_dataframe(self): 
        risk_list = []
        for value in self.COMPANY_LIST: 
            risk_list.append(self.return_risk(value))
        return pd.DataFrame(risk_list) 
    
    def jp(self,drawdown, skewness, kurtosis, value_at_risk,annualized_sharp_ratio,annualized_returns,annualized_volatility):
        RISK_MEASURE = []
        if drawdown: 
            RISK_MEASURE.append('drawdown')
        if skewness: 
            RISK_MEASURE.append('skewness')
        if kurtosis:
            RISK_MEASURE.append('kurtosis')
        if value_at_risk:
            RISK_MEASURE.append('value_at_risk')
        if annualized_sharp_ratio:
            RISK_MEASURE.append('annualized_sharp_ration')
        if annualized_returns:
            RISK_MEASURE.append('annualized_returns') 
        if annualized_volatility:
            RISK_MEASURE.append('annualize_volatility')
        
        return self.RISK_DATAFRAME.loc[:,RISK_MEASURE].iplot(kind = 'heatmap', dimensions = (900,400), colorscale = 'RdBu')
        print(RISK_MEASURE)
    
    def display_risk_measures(self):
        return widgets.interact(self.jp,
                     drawdown = self.a,
                     skewness = self.b,
                     kurtosis = self.c,
                     value_at_risk = self.d,
                     annualized_sharp_ratio = self.e,
                     annualized_returns = self.f,
                     annualized_volatility = self.g

                    )

class FitArima:
    def __init__(self, series_to_predict) -> None:
        self.close_series = series_to_predict 
        self.model = self.fit_autoarima()
    
    def perform_decomposition(self): 
        dec_object = seasonal_decompose(self.close_series,model = 'multiplicative',period = 256)
        df = pd.concat([dec_object.observed,dec_object.trend,dec_object.seasonal, dec_object.resid], axis = 1)
        df.iplot(subplots = True, theme = 'solar', colorscale = 'Set2', width = 1.5, dimensions = (1000,500)) 
        
    def fit_autoarima(self):

        seasonal_model = auto_arima(self.close_series,
                               start_p = 0,
                               start_q = 0,
                               test = 'adf',
                               max_p = 3,
                               max_q = 3,
                               start_P = 0,
                                start_Q=0, 
                                max_P=2, max_Q=2,
                                information_criterion='aic',
                               seasonal = True,
                               d = None,
                               trace = True,
                               D = 1,
                                n_jobs=4,
                             stepwise=False,
                            error_action = 'ignore',
                              )
        return seasonal_model 
    
    def plot_diagnosis(self): 
        plt.style.use('seaborn')
        print(self.model.summary())
        self.model.plot_diagnostics();
    
    def forecast_values(self,periods): 
        fc = self.model.predict(n_periods = periods)
        index_of_fc =pd.period_range(start = self.close_series.index[-1], periods = periods)
        fc_series = pd.Series(fc, index = index_of_fc) 
        plt.figure()
        fc_series.plot()
        self.close_series[-200:].plot()
    
    
    