import numba as nb 
import numpy as np 
import pandas as pd 
import math 

@nb.njit()
def ewma(x,alpha): 
    """
    x: value of series in the current time period
    alpha: The parameter decides how important 
    the current observation is in the calculation of ewma, the 
    higher value of alpha more closely ewma tracks the original 
    series 
    EWMA(t) = alpha * x(t) + (1 - alpha)*EWMA(t-1)
    """
    y = np.zeros_like(x) 
    y[0] = x[0] 
    for i in range(1, len(x)): 
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1] 
    return y 

@nb.njit()
def average_true_range(highs, lows, close, period = 14):
    alpha = 1/period 
    true_range = np.vstack((highs[1:] - lows[1:], 
               np.abs(highs[1:] - close[:-1]),
               np.abs(highs[1:] - close[:-1])
              ))#.max(axis = 0)
    
    max_true_range = np.empty(true_range.shape[1]) 
    for i in range(true_range.shape[1]): 
        max_true_range[i] = np.max(true_range[:,i])
    average_true_range = ewma(max_true_range,alpha)
    return np.concatenate((np.array([np.nan]),average_true_range)) 
    
@nb.njit()
def average_directional_index(highs, lows, close, periods = 14):
    alpha = 1 / periods
    dm_plus = highs[1:] - highs[:-1]
    dm_minus = lows[:-1] - lows[1:] 
    up_dm = np.where((dm_plus > dm_minus)&(dm_plus > 0),dm_plus,0.0)
    down_dm = np.where((dm_minus > dm_plus)&(dm_minus > 0), dm_minus,0.0) 
    smooth_up_dm = ewma(up_dm,alpha) 
    smooth_down_dm = ewma(down_dm, alpha) 
    smooth_up_dm_i = (smooth_up_dm*100) / average_true_range(highs, lows,close)[1:] 
    smooth_down_dm_i = (smooth_down_dm*100) / average_true_range(highs, lows,close)[1:] 
    directional_movement_index = (np.abs(smooth_up_dm_i - smooth_down_dm_i)*100) / np.abs(smooth_up_dm_i + smooth_down_dm_i)
    average_directional_movement_index = ewma(directional_movement_index,alpha)
    return average_directional_movement_index 

@nb.njit()
def simple_moving_average(array:np.array, window:int) -> np.array: 
    sma_array = np.zeros(array.shape)
    for i in range(0,len(array)-window): 
        sma_array[i+window] = np.sum(array[i:i+window])
    return sma_array/window 

@nb.njit()
def rsi(array, periods = 14, want_ewma = True):
    diff = np.diff(array)
# if difference is negative we have loss hence we make that loss zero 
    gains = np.where(diff < 0,0.0,diff)
    loss = np.abs(np.where(diff > 0,0.0,diff))
    if want_ewma == True: 
        moving_average_up = ewma(gains,1/periods) 
        moving_average_down = ewma(loss, 1/periods) 
    else:
        moving_average_up = simple_moving_average(gains,periods) 
        moving_average_down = simple_moving_average(loss, periods)  
    rsi = moving_average_up/moving_average_down 
    rsi = 100 - (100/(1+rsi))
    return rsi 
        
