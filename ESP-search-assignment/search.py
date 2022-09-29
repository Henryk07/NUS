# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

# Haolin Chen
# NUS ESP3201


from queue import *
import math
import time
from heapq import *
import copy
import heapq


class Stack:
    def __init__(self):
        self.content = []

    def push(self, item):
        self.content.insert(0, item)

    def pop(self):
        return self.content.pop(0)

    def size(self):
        return len(self.content)

    def clean(self):
        self.content = []


class Queue:
    def __init__(self):
        self.content = []

    def enqueue(self, item):
        self.content.insert(0, item)

    def dequeue(self):
        return self.content.pop()

    def size(self):
        return len(self.content)

    def clean(self):
        self.content = []


class PriorityQueue:

    def __init__(self):
        self._queue = []
        self._index = 0

    def insert(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def remove(self):
        return heapq.heappop(self._queue)[-1]

    def is_empty(self):
        return len(self._queue) == 0


class Status:
    def __init__(self, position, cost):
        self.position = position
        self.cost = cost


class Point:
    def __init__(self, w, cost, heuristic, parent, remaining):
        self.position = w
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent
        self.remaining = remaining


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "ucs": ucs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # BFS is Implemented with a queue

    return_path = []

    s = maze.getStart()

    # if start is the goal
    if maze.isObjective(s[0], s[1]):
        return_path.append(s)
        return return_path

    # queue for bfs
    queue = []
    queue.append(s)

    # set to keep track of visited
    visited = set()
    visited.add(s)

    # a map to keep track of the previous aka parent node
    prev = {}

    # bfs traversal
    while queue:
        s = queue.pop(0)
        if maze.isObjective(s[0], s[1]):
            return_path = [s]
            while return_path[-1] != maze.getStart():
                return_path.append(prev[return_path[-1]])
            return_path.reverse()
            return return_path

        neighbors = maze.getNeighbors(s[0], s[1])

        for i in neighbors:
            if i not in visited and i not in queue:
                prev[i] = s
                queue.append(i)
                visited.add(i)


def dfs(maze):
    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    discover = set()
    parent = {}
    path = []
    discover.add(maze.getStart())
    stack = Stack()
    stack.push(maze.getStart())
    while stack.size():
        v = stack.pop()
        for w in maze.getNeighbors(v[0], v[1]):
            if w not in discover:
                stack.push(w)
                discover.add(w)
                parent[w] = v
                if [w] == maze.getObjectives():
                    stack.clean()
                    break
    w = maze.getObjectives()[0]
    while True:
        path.append(w)
        w = parent[w]
        if w == maze.getStart():
            break
    path.append(maze.getStart())
    path = path[::-1]
    return path
    # return []


def ucs(maze):
    """
    Runs ucs for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    end = maze.getObjectives()[0]
    close = set()
    neighbors = {}
    parent = {}
    current_cost = {}
    path = []
    for i in range(maze.rows):
        for j in range(maze.cols):
            current_cost[(i, j)] = 9999
    current_cost[start] = 0
    heap = []
    heappush(heap, (current_cost[start], start))

    while len(heap):
        v = heappop(heap)
        close.add(v[1])
        if v[1] not in neighbors:
            neighbors[v[1]] = maze.getNeighbors(v[1][0], v[1][1])
        for w in neighbors[v[1]]:
            if w == end:
                parent[w] = v[1]
                heap = []
                break
            if w not in close:
                if current_cost[v[1]] + 1 < current_cost[w]:
                    current_cost[w] = current_cost[v[1]]+1
                    heappush(heap, (copy.deepcopy(
                        current_cost[w]), w))
                    parent[w] = v[1]
    w = end
    while True:
        path.append(w)
        w = parent[w]
        if w == start:
            break
    path.append(start)
    path = path[::-1]
    return path


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    end = maze.getObjectives()[0]
    return astar_calculator(maze, start, end)


def astar_calculator(maze, start, end):  # (Astar calc for astar and astar multi)
    close = set()
    neighbors = {}
    parent = {}
    current_cost = {}
    heuristic = {}
    path = []
    for i in range(maze.rows):
        for j in range(maze.cols):
            current_cost[(i, j)] = 9999
    current_cost[start] = 0
    heuristic[start] = abs(end[0]-start[0]) + abs(end[1]-start[0])
    heap = []
    heappush(heap, (current_cost[start] + heuristic[start], start))

    while len(heap):
        v = heappop(heap)
        close.add(v[1])
        if v[1] not in neighbors:
            neighbors[v[1]] = maze.getNeighbors(v[1][0], v[1][1])
        for w in neighbors[v[1]]:
            if w == end:
                parent[w] = v[1]
                heap = []
                break
            if w not in close:
                if current_cost[v[1]] + 1 < current_cost[w]:
                    current_cost[w] = current_cost[v[1]]+1
                    heuristic[w] = abs(end[0]-w[0]) + abs(end[1]-w[1])
                    heappush(heap, (copy.deepcopy(
                        current_cost[w])+copy.deepcopy(heuristic[w]), w))
                    parent[w] = v[1]
    w = end
    while True:
        path.append(w)
        w = parent[w]
        if w == start:
            break
    path.append(start)
    path = path[::-1]
    return path


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # just use the multi would be the same
    start = maze.getStart()
    objectives = maze.getObjectives()
    return astar_multi_calculator(maze, start, objectives)
    # return []


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    objectives = maze.getObjectives()
    return astar_multi_calculator(maze, start, objectives)


def astar_multi_calculator(maze, start, objectives):
    first = time.time()
    start = maze.getStart()
    objectives = maze.getObjectives()
    path = []
    map = {}
    MST_map = {}
    pair_map = {}
    for w in objectives:
        for m in objectives[objectives.index(w)+1:]:
            pair_map[(w, m)] = len(astar_calculator(copy.deepcopy(maze), w, m))
    for i in range(maze.rows):
        for j in range(maze.cols):
            s = Status((maze.rows, maze.cols), {})
            map[(i, j)] = s
    g = 0
    h = MST(maze, start, objectives, MST_map, pair_map)
    f = g+h
    p_start = Point(start, 0, f, None, objectives)
    map[p_start.position].cost[tuple(p_start.remaining)] = p_start.cost
    heap = []
    length = 99999
    destination = Point((), 0, 0, None, [])
    i = 0
    heappush(heap, (p_start.heuristic, i, p_start))
    while len(heap):
        v = heappop(heap)
        for w in maze.getNeighbors(v[2].position[0], v[2].position[1]):
            if w in v[2].remaining:
                remaining = copy.deepcopy(v[2].remaining)
                remaining.remove(w)
            else:
                remaining = copy.deepcopy(v[2].remaining)
            g = v[2].cost + 1
            h = MST(maze, w, remaining, MST_map, pair_map)
            f = g+h
            p = Point(w, g, f, v[2], remaining)
            if p.remaining == []:
                if length > p.cost:
                    length = p.cost
                    destination = p
                break
            if p.heuristic >= length:
                break
            if tuple(p.remaining) in map[p.position].cost:
                i += 1
                if map[p.position].cost[tuple(p.remaining)] > p.cost:
                    map[p.position].cost[tuple(p.remaining)] = p.cost
                    heappush(heap, (p.heuristic, i, p))
            if tuple(p.remaining) not in map[p.position].cost:
                i += 1
                heappush(heap, (p.heuristic, i, p))
                map[p.position].cost[tuple(p.remaining)] = p.cost
    p = destination
    while True:
        path.append(p.position)
        p = p.parent
        if p.parent == None:
            break
    path = path[::-1]
    path = [start] + path
    print(time.time()-first)
    print(path)
    return path


def find_root(current, parent_MST):
    w = current
    while True:
        w = parent_MST[w]
        if w == parent_MST[w]:
            return w


def isRoot(current, parent_MST):
    if current == find_root(current, parent_MST):
        return True
    else:
        return False


def Union(w, m, parent_MST):
    if find_root(w, parent_MST) == find_root(m, parent_MST):
        return 0
    elif isRoot(w, parent_MST) and not isRoot(m, parent_MST):
        parent_MST[find_root(m, parent_MST)] = copy.deepcopy(w)
        return 1
    elif isRoot(m, parent_MST) and not isRoot(w, parent_MST):
        parent_MST[find_root(w, parent_MST)] = copy.deepcopy(m)
        return 1
    else:
        parent_MST[find_root(w, parent_MST)] = copy.deepcopy(
            find_root(m, parent_MST))
        return 1


def MST(maze, current, objectives, MST_map, pair_map):
    # Minimum Spanning Tree
    parent_MST = {}
    d_list = []
    distance2 = {}
    edge = []
    sum = 0
    if objectives == []:
        return 0
    for w in objectives:
        d_list.append(len(astar_calculator(copy.deepcopy(maze), current, w)))
    d_list.sort()
    distance1 = d_list[0] - 1
    if tuple(objectives) in MST_map:
        return MST_map[tuple(objectives)] + distance1
    for w in objectives:
        parent_MST[w] = copy.deepcopy(w)
    if maze.rows*maze.cols > 100:
        for w in objectives:
            for m in objectives[objectives.index(w)+1:]:
                distance2 = pair_map[(w, m)]
                edge.append((distance2, (w, m)))
    else:
        for w in objectives:
            for m in objectives[objectives.index(w)+1:]:
                distance2 = abs(w[0]-m[0]) + abs(w[1]-m[1])
                edge.append((distance2, (w, m)))
    # if the maze is small, use Manhatten distance, if it's large, use A* distance.
    if edge == []:
        return distance1
    heapify(edge)
    start = heappop(edge)
    parent_MST[start[1][1]] = start[1][0]
    sum += start[0]
    while len(edge):
        v = heappop(edge)
        result = Union(v[1][0], v[1][1], parent_MST)
        if result != 0:
            sum += v[0]
    MST_map[tuple(objectives)] = sum
    return sum + distance1
