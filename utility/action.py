
import numpy as np

# original buy one stock once
# def buy(agent,stock_prices,t):
#     if agent.balance > stock_prices[t]:
#         agent.balance -= stock_prices[t]
#         agent.inventory.append(stock_prices[t])
#         reward=0
#         # return 'Buy: ${:.2f}'.format(stock_prices[t]), reward
#         return f'Buy: ${stock_prices[t]:.2f}', reward
#     else:
#         reward=0
#         return f'Buy: no cash to buy', reward    


# def sell(agent,stock_prices,t):
#     if len(agent.inventory) > 0:
#         agent.balance += stock_prices[t]
#         bought_price = agent.inventory.pop(0)
#         profit = stock_prices[t] - bought_price
#         # global reward
#         reward = profit
#         # return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit), reward
#         return f'Sell: ${stock_prices[t]:.2f} | Profit: ${profit:.2f}', reward
#     else:
#         reward=0
#         return f'Sell: No stock to sell', reward 

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

def hold_v2(agent,stock_prices,t,actions):
    reward =  0
    return 'Hold', reward

def hold_eval():
    return 'Hold'
    logging.info('Hold')