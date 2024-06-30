import gym
from gym import spaces
import numpy as np
import math
import torch
from sklearn.preprocessing import MinMaxScaler


class StockMarketEnv(gym.Env):
    def __init__(self, data, initial_capital=10000, history_length=12, timestep_prediction_length=1, device=torch.device("cpu")):
        super(StockMarketEnv, self).__init__()
        self.timestep_prediction_length = timestep_prediction_length
        self.data = data
        self.device = device
        self.history_length = history_length
        # define variable for holding information about capital in the proftolio
        self.initial_capital = initial_capital
        self.capital= initial_capital
        # define variable for holding portfolio allocations based on stocks
        
        # based on unique stock identifiers ([1:] to omit Date column)
        self.num_stocks = len(data.columns.get_level_values(0).unique()[1:])
        
        # base on unique feature column names such as "Close" etc.
        self.num_of_features = len(data.columns.get_level_values(1).unique()[1:])
        
        # Define observation space
        observable_features = self.num_stocks * self.num_of_features
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(observable_features,))
        
        # create a dataframe based on column in data 
        self.portfolio = np.zeros(self.num_stocks + 1)
        self.portfolio_ratios = [[0] * (self.num_stocks + 1)] # +1 to represent cash allocation

        self.portfolio_gains = []
        self.portfolio_gains_baseline = []

        # Define variales to track timestep, timestep starts from history_length since we want to look at history_length previous days
        self.num_timesteps = data.shape[0]
        self.timestep = self.history_length

        # we want to be able to allocate speicific amount of portfolio to each stock, later on we normalize so that the ratios sum up to 0
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks + 1,), dtype=np.float32)

        # Initialize state
        self.state = self.reset()


    def get_observation(self):
        """Returns an observation of the current state."""
        # current_prices = self.data.xs('Close', level=1, axis=1).iloc[self.timestep].values
        # get all columns per feature (Close Change SMA_50 SMA_200 RSI_14)
        current_market_data = self.data.iloc[self.timestep - self.history_length:self.timestep].values
        # remove timestamp
        current_market_data = current_market_data[:, 1:]        
        # concat with current portfolio values
        # observation = np.concatenate((current_market_data, self.portfolio))
        
        scaler = MinMaxScaler()
        current_market_data = scaler.fit_transform(current_market_data)
        observation = current_market_data.astype(np.float32)
        return observation


    def get_tickers(self):
        """Returns a list of tickers in the portfolio."""
        return self.data.columns.get_level_values(0).unique()[1:]


    def shuffle_tickers(self):
        """Shuffles the tickers in the portfolio."""
        ticker_columns = self.data.columns.get_level_values(0).unique()[1:]
        shuffled_tickers = np.random.permutation(ticker_columns)
        date = np.array(["Date"])
        shuffled_tickers = np.concatenate((date, shuffled_tickers))
        self.data = self.data[shuffled_tickers]
        return 1


    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        self.portfolio = np.zeros(self.num_stocks + 1)
        self.portfolio_ratios = []
        self.capital = self.initial_capital
        self.timestep = self.history_length

        info = {}
        return self.get_observation(), info


    def step(self, action):
        """Emulates a step in the environment, in our case a day in the market."""
        min_action = min(action)
        max_action = max(action)
        # normalize between 0 and 1
        if min_action < 0 or max_action > 1:
            action = [(action[i]) / (max_action - min_action) for i in range(len(action))]
        # make the sum up to 1
        action = [action[i] / sum(action) for i in range(len(action))]

        # Sanity check
        if round(sum(action), 8) != 1:
            raise Exception(f"Internal error, sum of all actions is not 1, sum of actions {action} = {sum(action)}")

        # if some action is negative we set it to 0
        if len(action) != self.num_stocks + 1:
            raise ValueError(f"Incorrect number of actions within the action vector. Given {len(action)} actions, expected {self.num_stocks + 1}")
        
        # get current portfolio value
        current_prices = self.data.xs('Close', level=1, axis=1).iloc[self.timestep].values
        # calculate portfolio value (portfolio allocations exluding cash)
        portfolio_value = np.sum(self.portfolio[1:] * current_prices) + self.capital


        # sell stocks if portfolio allocation lesser than current portfolio
        for i in range(1, len(action)):
            asset_value = current_prices[i - 1] * self.portfolio[i]
            # calculate amnount of stocks to sell (rounded down since we are not considering fractional stocks)
            # print(math.floor(action[i] * portfolio_value / current_prices[i]))
            # if action[i] == np.nan:
            #     action[i] = 0
            # print(action[i], portfolio_value, current_prices[i])
            new_number_of_stocks = int(math.floor(action[i] * portfolio_value / current_prices[i - 1]))
            if new_number_of_stocks < self.portfolio[i]:
                sold_assets_value = (self.portfolio[i] - new_number_of_stocks) * current_prices[i - 1]
                self.portfolio[i] = new_number_of_stocks
                self.capital += sold_assets_value
                
                
        # buy stocks if portfolio allocation greater than current portfolio
        for i in range(1, len(action)):
            asset_value = current_prices[i - 1] * self.portfolio[i]
            # calculate amnount of stocks to buy (rounded down since we are not considering fractional stocks)
            new_number_of_stocks = int(math.floor(action[i] * portfolio_value / current_prices[i - 1]))
            if new_number_of_stocks > self.portfolio[i]:
                # doublecheck if capital is sufficient
                bought_assets_value = (new_number_of_stocks - self.portfolio[i]) * current_prices[i - 1]
                if self.capital < bought_assets_value:
                    print(f"Portfolio: {self.portfolio}")
                    raise ValueError(f"Insufficient Capital, {self.capital} < {bought_assets_value}")
                self.capital -= bought_assets_value
                self.portfolio[i] = new_number_of_stocks

        # compute reward based on portfolio value next timestamp relative to current timestamp
        portfolio_value = np.sum(self.portfolio[1:] * current_prices) + self.capital
        next_prices = self.data.xs('Close', level=1, axis=1).iloc[self.timestep + 1].values
        portfolio_value_next = np.sum(self.portfolio[1:] * next_prices) + self.capital

        pct_changes_next = [(next - current)/current for current, next in zip(current_prices, next_prices)]

        pct_change_portfolio = ( (portfolio_value_next - portfolio_value) / portfolio_value )
        reward = pct_change_portfolio
        
        # compute baseline gain
        baseline = [1/self.num_stocks] * self.num_stocks
        baseline_value_next = np.sum(baseline * next_prices)
        pct_changes_baseline = [(next - current)/current for current, next in zip(current_prices, next_prices)]
        pct_change_baseline = np.mean(pct_changes_baseline)

        # update portfolio ratios, gain and abseline gain after successful asset reallocations for statistics
        self.portfolio_ratios.append(action)
        self.portfolio_gains.append(pct_change_portfolio)
        self.portfolio_gains_baseline.append(pct_changes_baseline)
        # increment timestep
        self.timestep += self.timestep_prediction_length

        # get next state
        next_state = self.get_observation()

        # update done state based on whether or not we have reached the end of the episode (we reach it once there is not enough timestep values left in future)
        done = self.timestep >= self.num_timesteps - self.timestep_prediction_length

        # Add truncated variable used to end episdoe earily, we dont use it ergo its always False
        truncated = False

        # Create info var for optional content (also not used)
        info = {}
        return next_state, reward, done, truncated, info

    def render(self):
        """Prints out information about the current state of environment."""
        current_prices = self.data.xs('Close', level=1, axis=1).iloc[self.timestep].values
        portfolio_value = np.sum(self.portfolio[1:] * current_prices)
        total_value = portfolio_value + self.capital
        print(f'Timestep: {self.timestep}')
        print(f'Prices: {current_prices}')
        print(f'Portfolio: {self.portfolio}')
        print(f'Portfolio ratios: {self.portfolio_ratios[-1]}')
        print(f'Capital: {self.capital}')
        print(f'Total Value: {total_value}')
        print(f"Increase: {(total_value - self.initial_capital) / self.initial_capital * 100:.2f}%")


    def evaluate_for_constant_allocation(self, action):
        """Evaluates portfolio performance for constant allocation (id est buy & hold)."""
        # normalize the action_space
        action = [action[i] / sum(action) for i in range(len(action))]

        current_prices = self.data.xs('Close', level=1, axis=1).iloc[self.timestep].values

        # get current portfolio value
        portfolio_value = np.sum(self.portfolio[:1] * current_prices) + self.capital

        # buy stocks if portfolio allocation greater than current portfolio
        for i in range(1, len(action)):
            asset_value = current_prices[i - 1] * self.portfolio[i]
            # calculate amnount of stocks to buy (rounded down since we are not considering fractional stocks)
            new_number_of_stocks = int(math.floor(action[i] * portfolio_value / current_prices[i - 1]))
            if new_number_of_stocks > self.portfolio[i]:
                # doublecheck if capital is sufficient
                bought_assets_value = (new_number_of_stocks - self.portfolio[i]) * current_prices[i - 1]
                if self.capital < bought_assets_value:
                    raise ValueError(f"Insufficient Capital, {bought_assets_value} < {self.capital}")
                self.capital -= bought_assets_value
                self.portfolio[i] = new_number_of_stocks

        portfolio_value = np.sum(self.portfolio[1:] * current_prices) + self.capital
        # get prcies at the end of data's timestamp
        end_prices = self.data.xs('Close', level=1, axis=1).iloc[-1].values
        portfolio_value_end = np.sum(self.portfolio[1:] * end_prices) + self.capital

        self.reset()

        print(f'Portfolio Value: {portfolio_value}, Portfolio Value End: {portfolio_value_end}')
        print(f'Percent change: {(portfolio_value_end - portfolio_value) / portfolio_value * 100:.2f}%')
        return portfolio_value, portfolio_value_end


    def get_historical_data(self):
        """Returns historical data of portfolio ratios that were assigned over ntire data set."""
        return self.portfolio_ratios


    def close(self):
        pass