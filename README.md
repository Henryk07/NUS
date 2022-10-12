# NUS ESP3201

for Machine Learning Purpose and testing as well. Everything will appear, just relax and try to fix it

live is live
no matter what kinds of live that is
still your live, just try to enjoy it, or it will help you to do so

# Run instruction

### For ANN assignment

```sh
python3 ANN_assignment/ANN_v6.py
```

## For P_Filter assignment

```sh
python3 P_Filter/ESP-particleFilter-assignment/main.py
```

## For ESP-search-assignment

### Requirements

python3
pygame

The main file to run the mp is mp1.py:

```sh
usage: mp1.py [-h] [--method {bfs,dfs,ucs,astar,astar_cornor,astar_multi}] [--scale SCALE]
              [--fps FPS] [--human] [--save SAVE]
              filename
```

examples:

```sh
python3 mp1.py bigMaze.txt --method dfs
python mp1.py tinySearch.txt --scale 30 --fps 10 --human
```

for help:

```sh
python mp1.py -h
```

## For ESP-reinforcement Learning assignment

Run manual control:

```sh
python gridworld.py -m
```

Run random agent:

```sh
python gridworld.py -a random
```

Run value agent with 10 iterations:

```sh
python gridworld.py -a value -i 10
```

Run q learning agent for 10 episodes:

```sh
python gridworld.py -a q --episodes 10
```

Run q learning agent with the update in place, under manual control:

```sh
python gridworld.py -a q ---episodes 10 -m
```
