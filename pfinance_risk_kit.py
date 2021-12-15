import pandas as pd 
import scipy.stats as st
import numpy as np 
import seaborn as sns 
from scipy.optimize import minimize 
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
            # floor value is dynamic it is always 80% of the previous peak if  
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
