from datetime import date
import pandas as pd 
import scipy.stats as st
import numpy as np 
import seaborn as sns 
from scipy.optimize import minimize 
import matplotlib.pyplot as plt 
import ipywidgets as widgets
import numba as nb
from IPython.display import display
def drawdown(return_series: pd.Series): 
    """Takes a time series of asset returns.
       returns a DataFrame with columns for 
       wealth index, previous peaks and 
       percentile drawdown
    """

    wealth_index = 1000 * (1 + return_series).cumprod() 
    previous_peaks = wealth_index.cummax() 
    drawdowns = (wealth_index - previous_peaks) / previous_peaks 
    return pd.DataFrame({
        'wealth': wealth_index,
        'previous_peaks': previous_peaks,
        'drawdown': drawdowns
    })

def get_ffme_returns(): 
    """
    Load the fama-french dataset for the returns of the top and bottom deciles by market_cap
    """
    me_m = pd.read_csv(r"D:\python-finance\coursera_finance\data\Portfolios_Formed_on_ME_monthly_EW.csv", 
    header = 0,
    index_col = 0,
    na_values = -99.99
    )  
    rets = me_m[['Lo 10','Hi 10']] 
    rets.columns = ['small_cap','large_cap'] 
    rets /= 100 
    rets.index = pd.to_datetime(rets.index, format='%Y%m')
    return rets 

def get_hfi_returns(): 
    """
    Load and format edhec hedge fund index returns 
    """
    hifi = pd.read_csv(r"D:\python-finance\coursera_finance\data\edhec-hedgefundindices.csv", 
    header = 0,
    index_col = 0,
    parse_dates= True 
    )  
    
    hifi /= 100 
    hifi.index.freq = 'M'
    return hifi

def skewness(r):
    """
    Alternative to scipy.stats.skew() 
    Computes the skewness of the supplied series or dataframe 
    returns a float or a Series
    """
    demeaned_r = r - r.mean() 
    sigma_r = r.std(ddof = 0) 
    exp = (demeaned_r ** 3).mean() 
    return exp / sigma_r ** 3 
00
def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis() 
    Computes the kurtosis of the supplied series or dataframe 
    returns a float or a Series
    """
    demeaned_r = r - r.mean() 
    sigma_r = r.std(ddof = 0) 
    exp = (demeaned_r ** 4).mean() 
    return exp / sigma_r ** 4

def semi_deviation(r): 
    """
    Returns the semideviation i.e standard deviation of the 
    negative values of returns. where r must me a Series or a 
    data frame
    """
    is_negative = r < 0 
    return r[is_negative].std(ddof = 0) 


def is_normal(r, level = 0.01):
    """
    Applies the jarque-Bera test to determine os a series is normal or not 
    This test is applied at 1% level by default. 
    Returns True of hypothesis normality is accepted, Flase otherwise

    H0: Skewness and Kurtosis follow a normal distribution
    """
    _, p_value = st.jarque_bera(r)

    return p_value > level 

def var_historic(r, level = 5): 
    """
    Returns the historic Value at Risk at a specified level 
    i.e. returns the number such that "level" percent of the returns
    fall below that number and the (100 - level) percent are above
    """
    # so there is 5% chance that in any given month we are going to loose about <8%> or worse 
    # for short selling 
    if isinstance(r, pd.DataFrame): 
        return r.apply(lambda x : np.percentile(x,5), axis = 0) * -1
    elif isinstance(r, pd.Series): 
        return np.percentile(r,5) * -1 
    else:
        raise TypeError('Expected Pandas Series or a DataFrame') 




def var_parametric_gaussian(r, level = 5): 
    """
    Returns the Parametruc Gaussian VaR of a Series or a DataFrame column 
    wise 
    """
    # compute the z-score assuming the distribution was normal with given 
    # and standard deviation 

    # so there is 5% chance that in any given month we are going to loose about <8%> or worse 
    # for short selling 
    level /= 100 
    if isinstance(r, pd.DataFrame): 
        return r.apply(lambda x : st.norm.ppf(level, x.mean(), x.std())) * -1 
    elif isinstance(r, pd.Series):
        return st.norm.ppf(level, r.mean(), r.std()) * -1 
    else:
        raise TypeError('Expected Pandas Series or a DataFrame')

def var_parametric_cornsih_fisher(r,level = 5):
    """
    Returns the modified VaR using Cornish-Fisher modification
    """
    # so there is 5% chance that in any given month we are going to loose about <8%> or worse 
    # for short selling 
    z = st.norm.ppf(level/100) 
    s = st.skew(r) 
    k = st.kurtosis(r) 
    z = (z +
            (z**2 - 1)*s/6 +
            (z**3 -3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
        )
    return -(r.mean() + z*r.std(ddof = 0))

def cvar_historic(r, level= 5): 
    """
    Computes the Conditional VaR of a Series or a DataFrame
    """
    # if thet five percent chance happens i.e worst 5% of the possible cases 
    # the average of that is <3.6%> loss 
    is_beyond = r <= var_historic(r, level = level) * -1 
    return r[is_beyond].mean() * -1 

def get_ind_returns():
    """
    Load and format ken french 30 induatry portfolios value weighted monthly returns
    """
    ind = pd.read_csv('D:/python-finance/coursera_finance/data/ind30_m_vw_rets.csv', header = 0, index_col = 0, parse_dates = True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns = map(str.strip, ind.columns)
    #ind.columns.str.strip() 
    return ind 


def get_ind_size():
    """
    Load and format ken french 30 industry portfolio sizes
    """
    ind = pd.read_csv('D:/python-finance/coursera_finance/data/ind30_m_size.csv', header = 0, index_col = 0, parse_dates = True)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns = map(str.strip, ind.columns)
    #ind.columns.str.strip() 
    return ind 


def get_ind_nfirms():
    """
    Load and format ken french 30 industry portfolio sizes
    """
    ind = pd.read_csv('D:/python-finance/coursera_finance/data/ind30_m_nfirms.csv', header = 0, index_col = 0, parse_dates = True) 
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period('M')
    ind.columns = map(str.strip, ind.columns)
    #ind.columns.str.strip() 
    return ind 



def annualize_returns(r, periods_per_year): 
    """
    Annualizes a set of returns:
    if this is monthly data then periods_per_year will be 12 
    if this is quarterly data the periods_per_year will be 4 

    annualized_return = (returns + 1).prod() ** (12/n_months) - 1
    """
    compounded_growth = (r + 1).prod() 
    n_periods = r.shape[0] # number of rows in the returns vector 
    return compounded_growth ** (periods_per_year / n_periods) - 1 


def annualize_volatility(r, periods_per_year): 
    """
    Annualized the volatility of a set of returns for 
    given periods per_year 
    """
    return r.std() * (periods_per_year ** 0.5) 

def sharpe_ratio(r, riskfree_rate, periods_per_year): 
    """
    Computes the annualized sharp ratio of a set of returns
    risk_free_rate = 0.03 
    sharp_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    Sharpe Ratio Grading Thresholds:
    Less than 1: Bad
    1 - 1.99: Adequate/good
    2 - 2.99: Very good
    Greater than 3: Excellent




    convert the annualized risk free rate to per period
    generally the risk free rate will be annualized so if 
    we want to compute sharp ratio with this risk-free rate 
    we need to convert the risk free date to per-period(the period given as 
    input to the function) 
    """
    riskfree_rate_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1 
    # we are subtracting our per period return with adjusted adjusted riskfree_rate_per_period 
    excess_return = r - riskfree_rate_per_period 
    annualized_excess_returns = annualize_returns(excess_return, periods_per_year) 
    annualized_volatility = annualize_volatility(r,periods_per_year) 
    return annualized_excess_returns / annualized_volatility 


def portfolio_return(weights, returns): 
    """
    Weights-> Returns
    """
    return weights.T @ returns 

def portfolio_volatility(weights, covmat): 
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights) ** 0.5 

def plot_efficient_frontier2(n_points, asset_returns,covmat): 
    """
    plot 2 asset efficient frontier 
    """
    if asset_returns.shape[0] != 2: 
        raise ValueError("plot_efficient_frontier2 can only plot 2-asset frontier") 

    weights = np.array([np.array([w, 1 - w]) for w in np.linspace(0,1,n_points)]) 
    portfolio_returns = np.array([portfolio_return(weight, asset_returns) for weight in weights]) 
    portfolio_volatilitys = np.array([portfolio_volatility(weight,covmat) for weight in weights]) 
    efficient_frontier = pd.DataFrame({
        'portfolio_returns': portfolio_returns,
        'portfolio_volatilitys': portfolio_volatilitys
    })
    return sns.scatterplot(x = 'portfolio_volatilitys', y = 'portfolio_returns', data = efficient_frontier)


def minimize_volatility(target_return,asset_returns, covmat):
    """
    I need you to get me this much <target_return> for least possible volatility 
    target_return -> weight_vector 
    """
    # expected return has as many rows as we have assets 
    n = asset_returns.shape[0] # these are number of assets returns that we want the weight for
    # providing the initial guess to the optimizer 
    initial_guess = np.repeat(1/n,n)  
    # providing constraints to the optimizer 
    # so that every weight should have some bounds 
    # and the optimizer of scipy requires requence of bounds for every weights 
    # [w1,w2,w3] --> [(0,1),(0,1),(0,1)] 
    bounds = tuple((0.0,1.0) for _ in range(n))
    # bounds = ((0.0,1.0),) * n 
    # we have to make sure that whatever weights the optimizer comes up with 
    # has to satisfy the constraints of returns 
    # the function inside the constraints whill accept the 
    # parameter of weights 
    
    # CONSTRAINT 1 
    # the optimizer generally accepts only single argument (in this case it is weights) but we are passing a extra argument called asset_return
    return_is_target = {
        'type': 'eq',
        'args': (asset_returns,), 
        'fun': lambda weights, asset_returns:  target_return - portfolio_return(weights, asset_returns) # the portfolio returns for those weights is equal to the target return 
        
    }
    # CONSTRAINT 2 
    weights_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1 
    }
    
    
    
    # initial_guess: this is initial guess of weight we can think of passing this initial guess as argument to the optimizer 
            #function and if we want to specify anything except the initial guess we are going to pass it into the args parameter 
    # args : portfolio_volatility requires 2 params weights and
            #covariance matrix so we are passing the second argument for function to minimize (portfolio_volatility the vovariance matrix)
    
    results = minimize(portfolio_volatility,
                                 initial_guess, 
                                 args = (covmat,), 
                                 method = "SLSQP",
                                 #options = {'disp': False}
                                 constraints=(return_is_target, weights_sum_to_one),
                                 bounds=bounds
                                )
    return results.x  
    
    
def optimal_weights(n_points, asset_returns, covmat): 
    """
    List of returns to run the optimizer on to minimize the volatility
    """
    # all we have to do is generate a list of target_returns and send that to the optimizer
    # which already knows how to optimize weights for given target_returns 
    
    target_returns = np.linspace(asset_returns.min(), asset_returns.max(), n_points) 
    weights = [minimize_volatility(target_return,asset_returns,covmat) for target_return in target_returns] 
    return np.array(weights)
    

def maximum_sharp_ratio(asset_returns, covmat, riskfree_rate):
    """
   This function will give best possible sharp ratio
    """
    # expected return has as many rows as we have assets 
    n = asset_returns.shape[0] # these are number of assets returns that we want the weight for
    # providing the initial guess to the optimizer 
    initial_guess = np.repeat(1/n,n)  

    bounds = tuple((0.0,1.0) for _ in range(n))
 
  
    # CONSTRAINT 2 
    weights_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1 
    }
    
    def neg_sharp_ratio(weights, riskfree_rate, asset_returns, covmat):
        """
        Returns the negative sharp ratio for given weights
        """
        returns = portfolio_return(weights, asset_returns)
        volatility = portfolio_volatility(weights, covmat) 
        sharp_ratio =  (returns - riskfree_rate) / volatility
        return -1 * sharp_ratio 
        
        
    results = minimize(neg_sharp_ratio,
                                 initial_guess, 
                                 args = (riskfree_rate,asset_returns,covmat), 
                                 method = "SLSQP",
                                 #options = {'disp': False}
                                 constraints=(weights_sum_to_one),
                                 bounds=bounds
                                )
    return results.x 


def global_minimum_variance(covmat):
    """
    if there is a situation where all the portfolio expected returns are same so the optimizer is going to 
    optimize the max sharp ratio by only decreasing the volatility. 
    `This is because the we are optimizing for shapr ratio which has the defination of <returns> / <volatility>
    and be it any combination of weights they are going to sum to 1 so the <returns> remains costant to the way 
    optimizer is going to find the max-sharp ratio is by minimizing the volatility
    `
    """
    n = covmat.shape[0]

    return maximum_sharp_ratio(riskfree_rate=0,covmat=covmat,asset_returns=np.repeat(1,n))

def plot_efficient_frontier(n_points, asset_returns,covmat, show_capital_market_line = False, style = '.-', riskfree_rate = 0, show_equal_weights = False, show_global_minimum_variance = False): 
    """
    plot asset efficient frontier 
    """
   
    weights =  optimal_weights(n_points, asset_returns, covmat)
    portfolio_returns = np.array([portfolio_return(weight, asset_returns) for weight in weights]) 
    portfolio_volatilitys = np.array([portfolio_volatility(weight,covmat) for weight in weights]) 

    efficient_frontier = pd.DataFrame({
        'portfolio_returns': portfolio_returns,
        'portfolio_volatilitys': portfolio_volatilitys
    })
    ax = sns.scatterplot(x = 'portfolio_volatilitys', y = 'portfolio_returns', data = efficient_frontier) 
    # get the weights of maximum sharp ratio 

    if show_global_minimum_variance:
        n = asset_returns.shape[0] 
        weights_gmv = global_minimum_variance(covmat) 
        returns_gmv = portfolio_return(weights_gmv,asset_returns)
        volatility_gmv = portfolio_volatility(weights_gmv, covmat) 
        ax.plot([volatility_gmv], [returns_gmv], marker = 'o', linewidth = 12, color = 'goldenrod')
    
    if show_equal_weights:
        n = asset_returns.shape[0] 
        equal_weights = np.repeat(1/n,n) 
        returns_equal_weights = portfolio_return(equal_weights,asset_returns)
        volatility_equal_weights = portfolio_volatility(equal_weights, covmat) 
        ax.plot([volatility_equal_weights], [returns_equal_weights], marker = 'o', linewidth = 12, color = 'red')


    if show_capital_market_line:
        weights_maximum_sharp_ratio = maximum_sharp_ratio(asset_returns,covmat,riskfree_rate)
        returns_max_sharp_ratio = portfolio_return(weights_maximum_sharp_ratio,asset_returns)
        volatility_max_sharp_ratio = portfolio_volatility(weights_maximum_sharp_ratio, covmat) 
        # the x points of cml is 0 because risk free rate has zero volatility and the second x point if volatility of maximum sharp ratio  
        cml_x = [0,volatility_max_sharp_ratio] 
        # the y values are the risk free rate and returns of weights of maximum sharp ratio
        cml_y = [riskfree_rate, returns_max_sharp_ratio] 
        ax.plot(cml_x, cml_y, color = 'green', marker = 'o', linestyle = 'dashed', linewidth=3)
    return ax 


def run_cppi(risky_r, safe_r = None, m = 3, start = 1000, floor = 0.8, riskfree_rate = 0.03, drawdown = None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # Set Up CPPI parameters 
    dates = risky_r.index 
    n_steps = len(dates) 
    account_value = start 
    floor_value = start * floor 
    peak = start 

    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns = ['R']) 
    
    if safe_r is None: 
        safe_r = pd.DataFrame().reindex_like(risky_r) 
        safe_r.values[:] = riskfree_rate / 12 
        # 
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_weights_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r) 

    for step in range(n_steps): 
        if drawdown is not None:
            # floor value is dynamic it is always 80% of the previous peak
            # we do not want to keep 80% of our wealth 
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)

        cushion = (account_value - floor_value) / account_value
        risky_weight = m * cushion 
        risky_weight = np.minimum(risky_weight,1) 
        risky_weight = np.maximum(risky_weight,0)
        safe_weight = 1 - risky_weight 
        risky_allocation = account_value * risky_weight 
        safe_allocation = account_value * safe_weight 

        account_value = risky_allocation * (1 + risky_r.iloc[step]) + safe_allocation * (1 + safe_r.iloc[step]) 

        cushion_history.iloc[step] = cushion 
        risky_weights_history.iloc[step] = risky_weight 
        account_history.iloc[step] = account_value 
    risky_wealth = start * (1 + risky_r).cumprod() 

    backtest_result = {
        'cppi_wealth': account_history, 
        'risky_wealth': risky_wealth,
        'risk_budget': cushion_history,
        'risky_allocation': risky_weights_history,
        'm': m,
        'start': start,
        'floor': floor, 
        'risky_r': risky_r,
        'safe_r': safe_r
    }

    return backtest_result 

def summary_stats(r: pd.Series, riskfree_rate = 0.03): 
    """
    Return a DataFrame that contaons aggregated summary stats for the returns 
    in the column r 
    """
    annualized_returns = annualize_returns(r, periods_per_year=12)
    annualized_volatility = annualize_volatility(r, periods_per_year=12) 
    annualized_sharpratio = sharpe_ratio(r,riskfree_rate,12) 
    maximum_drawdown = drawdown(r)['drawdown'].min() 
    skewness_ = skewness(r)  
    kurtosis_ = kurtosis(r) 
    value_at_risk_historic = var_historic(r)
    value_at_risk_parametric = var_parametric_cornsih_fisher(r) 
    
    return pd.DataFrame({
        'annualized_returns': annualized_returns,
        'annualized_volatility': annualized_volatility,
        'annualized_sharpratio': annualized_sharpratio,
        'maximum_drawdown': maximum_drawdown,
        'skewness': skewness_,
        'kurtosis': kurtosis_,
        'value_at_risk_historic': value_at_risk_historic,
        'value_at_risk_parametric': value_at_risk_parametric
    }, index = [0]).T

def geometric_brownian_motion(num_years = 10, n_senarios = 1000, mu = 0.07, sigma = 0.15, steps_per_year = 12, initial_price = 100, prices = True):
    """
    Evolution of Geomrtric Brownain Motion trajectories, sych for a stock price through a monte-carlo simulations 
    :param num_years: The number of years to generate data for
    :param n_senarios: The number of senarios/trajectories 
    :param mu: Annualized drift 
    :param sigma: annualized volatility 
    :param steps_per_year: granularity of the simulations 
    :param initial_price: initial price
    :param prices: return prices values of return values
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows

    """
    dt = 1 / steps_per_year 
    n_steps = int(num_years * steps_per_year) 
    rets_plus_one = np.random.normal(loc = (1 + mu*dt), scale = sigma * np.sqrt(dt), size = (n_steps, n_senarios)) 
    prices = pd.DataFrame(rets_plus_one).cumprod() * initial_price if prices else rets_plus_one - 1 
    return prices 


def show_cppi(n_senarios = 50, mu = 0.07, sigma = 0.15, m = 3, floor = 0.8, riskfree_rate = 0.03, y_max = 100, steps_per_year = 12):
    """
    Plot results of Monte Carlo simulations of CPPI
    """
    start = 100 
    simulated_returns = geometric_brownian_motion(n_senarios=n_senarios,
                                                      mu = mu, 
                                                      sigma = sigma,
                                                      steps_per_year=steps_per_year,
                                                      prices = False 
                                                     )
    risky_returns = pd.DataFrame(simulated_returns)
    # run the cppi back test 
    back_test_results = run_cppi(risky_returns,
                                     m = m, 
                                     riskfree_rate=riskfree_rate,
                                     start = start,
                                     floor = floor 
                                    )
    wealth = back_test_results['cppi_wealth'] 
    y_max = wealth.values.max() * y_max / 100 
    # this is the final wealth generated form each simulations 
    terminal_wealth = wealth.iloc[-1]
    terminal_wealth_mean = terminal_wealth.mean() 
    terminal_wealth_median = terminal_wealth.median() 
    
    failure_mask = terminal_wealth < floor * start 
    number_failures = np.sum(failure_mask)
    percentage_failures = number_failures / n_senarios 
    # expected shortfall: when there is a failure what is the average failure extent 
    # if we lost money below the floor value how much actually did we loose 
    expected_shortfall = (np.dot(failure_mask, terminal_wealth - start * floor) / number_failures) if number_failures > 0 else 0.0 
    
    
    
    
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, dpi = 130)  
    wealth.plot(legend = False, alpha = 0.3, color = 'steelblue', figsize = (11,5), ax = wealth_ax)
    wealth_ax.axhline(y = start, ls = ':', color = 'black') 
    wealth_ax.axhline(y = start * floor, ls = '--', color = 'red')
    wealth_ax.set_ylim(top = y_max)
    wealth_ax.set_ylabel('cppi_wealth')
    wealth_ax.set_title('simulations')
    
    terminal_wealth.plot.hist(ax = hist_ax, bins = 50, fc = 'darkorange', orientation = 'horizontal', alpha = 0.8) 
    hist_ax.axhline(y = start, ls = ':', color = 'black', label = 'start_price')
    hist_ax.axhline(y = terminal_wealth_mean, ls = '--', color = 'purple', label = 'mean')
    hist_ax.axhline(y = terminal_wealth_median, ls = '--', color = 'blue', label = 'median')
    hist_ax.set_title('Terminal Wealth Distribution')
    
    #wealth_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    hist_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    hist_ax.annotate(f"Mean: {int(terminal_wealth_mean)}", xy = (1.01,0.9), xycoords = 'axes fraction')
    hist_ax.annotate(f"Median: {int(terminal_wealth_median)}", xy = (1.01,0.85), xycoords = 'axes fraction')
    
    if floor > 0.01: 
        hist_ax.axhline(y = start * floor, ls = '--', color = 'red')
        hist_ax.annotate(f"Violations: {number_failures} \n\nShortfall = ${expected_shortfall}", xy = (0.7,0.7), xycoords = 'axes fraction')
    plt.tight_layout()
    
def display_cppi(show_cppi = show_cppi):
    """
    uses the above defined function to gain interactivity to the cppi-backtesting
    """
    cppi_controls = widgets.interactive(
    show_cppi,
    n_senarios = widgets.IntSlider(min = 1, max = 1000, step = 5, value = 50),
    mu = widgets.FloatSlider(min = 0.0, max = 0.2, step = 0.01,value = 0.07),
    sigma = widgets.FloatSlider(min = 0.0, max = 0.3, step = 0.05, value = 0.15),
    floor = widgets.FloatSlider(min = 0, max = 1, step = 0.1, value = 0.8),
    m = widgets.FloatSlider(min = 1, max = 5, step = 0.5, value = 2), 
    riskfree_rate = widgets.FloatSlider(min = 0.0, max = 0.05, step = 0.01, value = 0.03), 
    y_max = widgets.IntSlider(min = 0, max = 100, step = 1, value = 100, description = 'Zoom Y Axis'),
    steps_per_year = widgets.IntSlider(min = 1, max = 12, value = 12))

    display(cppi_controls)


def discount(time_period, rate_of_intrest): 
    """
    Compute the price of a pure discount bond that pays 
    1 unit at time at time_period where time_period is in years 
    and rate_of_intrest is the annual intrest date 

    simply saying what is value of 1 unit currency `time_period` years from now.
    """

    return (1 + rate_of_intrest) ** (-time_period)

def present_value(liabilities, rate_of_intrest): 
    """
    Compute the present value of a list of liabilities given by the time (as index) and amounts as values
    """
    dates = liabilities.index 
    discounted_liabilities = discount(dates, rate_of_intrest)  
    return (discounted_liabilities * liabilities).sum() 

def funding_ratio(assets, liabilities, rate_of_intrest): 
    """
    Computes the funding ratio of a series of liabilities, based on an intrest rate and current value of assets

    if funding_ratio = 0.8 how much money do we have compared to how much money we need hence 
    we have 0.8 money but we need 1 
    """
    return assets / present_value(liabilities, rate_of_intrest) 



import math 
@nb.njit()
def instantenous_to_annual_rate(short_rate_of_intrest): 
    """
    Converts short rate to annualized rate of intrest 
    r_annual = exp(short_rate_of_intrest) - 1 
    """
    return np.expm1(short_rate_of_intrest)
@nb.njit()
def annual_to_instantenous_rate(annual_rate_of_intrest): 
    """
    Convert annualized rate to a short-rate 
    """
    return np.log1p(annual_rate_of_intrest) 
@nb.njit()
def cir_model_numba(num_years = 10, num_senarios = 1, a = 0.05, b = 0.03, sigma = 0.05, steps_per_year = 12, initial_intrest_rate = 0.0): 
    """
    Implements the cir model 
    :param num_years: Number of years to generate the data for
    :param num_senarios: Number of senarios
    :param a: rate of mean reversion
    :param b: long term mean of intrest rate
    :param sigma: volatility 
    :steps_per_year: granularity of simulations
    :param initial_intrest_rate: Intrest rate before the simulations start ,if initial intrest rate is not mentioned set the initial intrest rate long term mean of intrest
    """
    # if initial intrest rate is not mentioned set the initial intrest rate long term mean of intrest 
    if initial_intrest_rate == 0.0: initial_intrest_rate = b 
    # this intrest rate should be converted into a instantenous rate 
    # because cir model only works only with instantenous intrest rate at time t 
    initial_intrest_rate = annual_to_instantenous_rate(initial_intrest_rate) 
    
    dt = 1 / steps_per_year 
    # because i am going to initialize that array of rates and the rates are 
    # going to contain initial rate at row zero so i need one more time step 
    num_steps = int(num_years * steps_per_year) + 1 
    shock = np.random.normal(0, scale = np.sqrt(dt), size = (num_steps,num_senarios)) 
    rates = np.empty_like(shock)
    # rates is an empty array with initial value set from the initial_intrest_rate 
    # and progressively we will be calculating the rate based in last relative rate 
    rates[0] = initial_intrest_rate 
    # the looping starts at 1 because we have already filled the 
    # initial rate 
    for step in range(1, num_steps): 
        # r_t = instantanous intrest rate at time t 
        # we are referencing the rate from rates array 
        # at step 1 we will be looking rates from step zero to calculate the next rate 
        # and to this equation we will be using the shock term 
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) + sigma * np.sqrt(r_t)*shock[step]
        # we have found the difference of rates hence we can update the next rate 
        rates[step] = np.abs(r_t + d_r_t)
    return instantenous_to_annual_rate(rates) 


def show_cir(initial_intrest_rate = 0.03, a = 0.5, b = 0.03, sigma = 0.05, num_senarios = 5): 
    pd.DataFrame(cir_model_numba(initial_intrest_rate=initial_intrest_rate,b = b, a = a, sigma=sigma,num_senarios = num_senarios)).iplot(theme = 'solar',  dimensions = (1200,700)
                                                                                                                                        )


def display_cir():
    controls = widgets.interact(show_cir,
                                initial_intrest_rate = widgets.FloatSlider(min = 0, max = 0.15, step = 0.01, value = 0.02),
                                a = widgets.FloatSlider(min = 0, max = 1, step = 0.1, value = 0.5),
                                b = widgets.FloatSlider(min = 0, max = 0.15, step = 0.01, value = 0.03),
                                sigma = widgets.FloatSlider(min = 0, max = 0.1, step = 0.01, value = 0.05),
                                num_senarios = widgets.IntSlider(min = 1, max = 100, step = 5, value = 10)
                            )
    return controls 
                                                               




#@njit(fastmath = True)
@nb.njit()
def price_bond(ttm, r,h,a,b,sigma):
    """
    :param ttm: T - t where T is maturity and t is time step `t`
    :param r: rate of intrest at time step t
    """
    _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
    _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
    _P = _A*np.exp(-_B*r)
    return _P
    
#@njit(parallel = True, fastmath = True, nogil = True)
@nb.njit()
def cir_model_bond_returns(num_years = 10, num_senarios = 1, a = 0.05, b = 0.03, sigma = 0.05, steps_per_year = 12, initial_intrest_rate = 0.0):
    """
    Implements the cir model 
    :param num_years: Number of years to generate the data for
    :param num_senarios: Number of senarios
    :param a: rate of mean reversion
    :param b: long term mean of intrest rate
    :param sigma: volatility 
    :steps_per_year: granularity of simulations
    :param initial_intrest_rate: Intrest rate before the simulations start ,if initial intrest rate is not mentioned set the initial intrest rate long term mean of intrest
    """

    if initial_intrest_rate  == 0.0: initial_intrest_rate = b 
    initial_intrest_rate = annual_to_instantenous_rate(initial_intrest_rate) 
    dt = 1 / steps_per_year 
    num_steps = int(num_years * steps_per_year) + 1 
    shock = np.random.normal(0, scale = np.sqrt(dt), size = (num_steps, num_senarios)) 
    rates = np.empty_like(shock) 
    rates[0] = initial_intrest_rate 
#     @njit()
#     def price_bond(ttm, r):
#         """
#         :param ttm: T - t where T is maturity and t is time step `t`
#         :param r: rate of intrest at time step t
#         """
#         _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
#         _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
#         _P = _A*np.exp(-_B*r)
#         return _P
    ### For price generation 
    h = math.sqrt(a**2 + 2*sigma**2) 
    prices = np.empty_like(shock) 
    # num_years: bond maturity 
    # 0: initial no time has passed 
    prices[0] = price_bond(num_years,initial_intrest_rate,h,a,b,sigma) 
    ### 
    # the looping starts at 1 because we have we have already initialized price and intrest rate 
    for step in range(1, num_steps): 
        r_t = rates[step - 1] 
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step] 
        rates[step] = np.abs(r_t + d_r_t) 
        prices[step] = price_bond(num_years - step*dt,rates[step],h,a,b,sigma)
    return instantenous_to_annual_rate(rates), prices 


def show_cir_prices(initial_intrest_rate = 0.03, a = 0.5, b = 0.03, sigma = 0.05, num_senarios = 5): 
    rates, prices = cir_model_bond_returns(num_years = 10, num_senarios = num_senarios, a = a, b = b, sigma = sigma, steps_per_year = 12, initial_intrest_rate = initial_intrest_rate)
    pd.DataFrame(prices).iplot(theme = 'solar', dimensions = (1200,700))                                                                     

def display_cir_prices():
    controls = widgets.interact(show_cir_prices,
                            initial_intrest_rate = widgets.FloatSlider(min = 0, max = 0.15, step = 0.01, value = 0.03),
                            a = widgets.FloatSlider(min = 0, max = 1, step = 0.1, value = 0.5),
                            b = widgets.FloatSlider(min = 0, max = 0.15, step = 0.01, value = 0.03),
                            sigma = widgets.FloatSlider(min = 0, max = 0.1, step = 0.01, value = 0.05),
                            num_senarios = widgets.IntSlider(min = 1, max = 200, step = 5, value = 10)
                           )
    return controls 
