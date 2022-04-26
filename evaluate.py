import argparse
import importlib
import logging
import sys
import os
import numpy as np
# np.random.seed(3)  # for reproducible Keras operations

from utility.action import *
from utility.utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='0050_2014', help="stock name")
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

model_to_load = inputs.model_to_load
model_name = model_to_load.split('_')[0]
stock_name = inputs.stock_name
stock_margin = stock_margins(stock_name)
initial_balance = inputs.initial_balance
display = True
window_size = 10
action_dict = {0: 'Hold', 1: 'Hold', 2: 'Sell'}

# select evaluation model
model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=13+3, balance=initial_balance, is_eval=True, model_name=model_name)

def buy(agent,stock_prices,t):
    if agent.balance > stock_prices[t]:
        # buy maximum stocks 
        buy_num = int(agent.balance/stock_prices[t])
        agent.balance -= stock_prices[t] * buy_num
        agent.inventory.extend([stock_prices[t]]*buy_num)
        agent.buy_dates.append(t)
        reward=0
        return f'Buy: ${stock_prices[t]:.2f}, {buy_num} stocks | Reward: ${reward:.2f}', reward
    else:
        reward=0
        return f'Buy: no cash to buy | Reward: ${reward:.2f}', reward

def sell(agent,stock_prices,t):
    if len(agent.inventory) > 0:
        sell_num = len(agent.inventory) #持有股數
        sell_balance = stock_prices[t] * sell_num # 當日賣出總額
        bought_balance = agent.inventory[0] * sell_num # 當初購買價格
        reward = sell_balance - bought_balance
        agent.balance += sell_balance
        agent.inventory = []
        agent.sell_dates.append(t)
        return f'Sell: ${stock_prices[t]:.2f}, {sell_num} stocks | Reward: ${reward:.2f}', reward
    else:
        reward=0
        return f'Sell: No stocks to sell | Reward: ${reward:.2f}', reward

def hold_eval():
    return 'Hold'

# configure logging
logging.basicConfig(filename=f'logs/{model_name}_evaluation_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

portfolio_return = 0
while portfolio_return == 0: # a hack to avoid stationary case
    stock_prices = stock_close_prices(stock_name)
    trading_period = len(stock_prices) - 1
    state = generate_combined_state(0, window_size, stock_prices, stock_margin, agent.balance, len(agent.inventory)) 
    for t in range(1, trading_period + 1):
        if model_name == 'DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
            actions = agent.model.predict(state)[0]
            action = agent.act(state)

        # print(state)
        # print('actions:', action, actions)

        next_state = generate_combined_state(t, window_size, stock_prices, stock_margin, agent.balance, len(agent.inventory)) 
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

        execution_result = f'{action} None'
        # execute position
        logging.info(f'Step: {t}')
        if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0: 
            execution_result = hold_eval() # hold
        if action == 1 and agent.balance > stock_prices[t]: 
            execution_result, reward, agent = buy(agent,stock_prices,t)  # buy
        if action == 2 and len(agent.inventory) > 0: 
            execution_result, reward, agent = sell(agent,stock_prices,t) # sell

        logging.info(execution_result)    
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        agent.portfolio_values.append(current_portfolio_value)
        state = next_state

        done = True if t == trading_period else False
        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)

if display:
    # plot_portfolio_transaction_history(stock_name, agent)
    # plot_portfolio_performance_comparison(stock_name, agent)
    plot_all(stock_name, agent)
