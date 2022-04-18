
import numpy as np

def buy(agent,stock_prices,t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        reward=0
        # return 'Buy: ${:.2f}'.format(stock_prices[t]), reward
        return f'Buy: ${stock_prices[t]:.2f}', reward
    else:
        reward=0
        return f'Buy: no cash to buy', reward    

def sell(agent,stock_prices,t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        # global reward
        reward = profit
        # return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit), reward
        return f'Sell: ${stock_prices[t]:.2f} | Profit: ${profit:.2f}', reward
    else:
        reward=0
        return f'Sell: No stock to sell', reward 

def hold(agent,stock_prices,t,actions):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(agent,t)
            actions[next_probable_action] = 1 # reset this action's value to the highest
            return 'Hold', actions
