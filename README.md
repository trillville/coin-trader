# Coin Trader
Coin trading using reinforcement learning!

## Agent and Model overview
Environment <-> Runner <-> Agent <-> Model

## Some helpful reference documents: 
- [minimal test environment](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/environments/minimal_test.py)
- [TensorForce blog](https://reinforce.io/blog/introduction-to-tensorforce/)
- [TensorForce manual](https://media.readthedocs.org/pdf/tensorforce/latest/tensorforce.pdf)
- [TensorForce reference docs](http://tensorforce.readthedocs.io/en/latest/agents_models.html)

## Development ideas
- Need to set up a test/train environment (incl. implementation of RandomAgent) to start producing baselines
- Different/better objective function (e.g. Sharpe ratio) that:
  - penalizes excessive trading
  - high variance in returns
  - long periods of inactivity
- Network architecture:
  - CNN -> May be good way to process lots of noisy data
  - RNN -> LSTM is the standard approach to timeseries data
  - should agent state + environment state be included in the network archicture for the entire funnel? Intuitively, shove in agent state at the very end (post timeseries)
- Code is very messy right now and i used a bajillion self. references to make things work... needs major refactor
- Volume of trading data (1k, 10k, 100k, 1M record datasets would be ideal)
  
