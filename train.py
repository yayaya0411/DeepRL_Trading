import argparse
import importlib
import logging
import sys
import time
import tqdm

# from utility.action import *
from utility.utils import *


pd.set_option('display.float_format', lambda x: '%.3f' % x)

parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='0050_2013', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=100, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance

stock_prices = stock_close_prices(stock_name)
stock_margin = stock_margins(stock_name)
trading_period = len(stock_prices) - 1  # 訓練期間，input stock data的總日期
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

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

def hold(agent,stock_prices,t,actions):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(agent,stock_prices,t)
            actions[next_probable_action] = 1 # reset this action's value to the highest
            return 'Hold', actions


# configure logging
logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Trading Object:           {stock_name}')
logging.info(f'Trading Period:           {trading_period} days')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

# select learning model
model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=13 + 3, balance=initial_balance,model_name=model_name)

start_time = time.time()
for e in tqdm.tqdm(range(1, num_episode + 1)):
    logging.info(f'\nEpisode: {e}/{num_episode}')

    agent.reset() # reset to initial balance and hyperparameters
    state = generate_combined_state(0, window_size, stock_prices, stock_margin, agent.balance, len(agent.inventory)) 
    # 將prince_state與portfolio_state 橫向串連起來橫向串連，作為input state

    for t in range(1, trading_period + 1):
        if t % 1000 == 0:
            logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')
        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, stock_margin, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        
        if model_name == 'DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
            actions = agent.model.predict(state)[0]
            action = agent.act(state)
        # execute position
        logging.info('Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1], actions[2]))
        if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0: # hold
            execution_result = hold(agent,stock_prices,t,actions)
        if action == 1: # buy
            execution_result, reward = buy(agent,stock_prices,t)      
        if action == 2: # sell
            execution_result, reward = sell(agent,stock_prices,t)        
        
        # check execution result
        if execution_result is None:
            reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
        else:
            if isinstance(execution_result, tuple): # if execution_result is 'Hold'
                execution_result = execution_result[0]
                actions = execution_result[1]
            logging.info(execution_result)    
                        
        # calculate reward
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        # reward += unrealized_profit

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)

        # update state
        state = next_state

        # experience replay
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            loss,mini_batch = agent.experience_replay()
            # print(t,len(mini_batch))
            # logging.info(f'Episode: {e}\tLoss: {loss:.2f}\tAction: {action_dict[action]}\tReward: {reward:.2f}\tBalance: {agent.balance:.2f}\tNumber of Stocks: {len(agent.inventory)}'.format(e, loss, action_dict[action], reward, agent.balance, len(agent.inventory)))
            logging.info(f'Episode: {e}\tLoss: {loss:.2f}\tAction: {action_dict[action]}\tReward: {reward:.2f}\tBalance: {agent.balance:.2f}\tNumber of Stocks: {len(agent.inventory)}')
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value})

        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            returns_across_episodes.append(portfolio_return)

    # save models periodically
    if e % 20 == 0:
        if model_name == 'DQN':
            agent.model.save(os.path.join(f'saved_models',f'{model_name}_{agent.state_dim}_dim_ep{e}.h5'))
        elif model_name == 'DDQN':
            agent.model.save('saved_models/DDQN_ep' + str(e) + '.h5')
            agent.model_target.save('saved_models/DDQN_ep' + str(e) + '_target.h5')
        elif model_name == 'DDPG':
            agent.actor.model.save_weights('saved_models/DDPG_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/DDPG_ep{}_critic.h5'.format(str(e)))
        logging.info('model saved')

logging.info('total training time: {0:.2f} min'.format((time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)
