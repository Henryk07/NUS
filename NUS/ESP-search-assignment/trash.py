''' UCS backup 
node = Nodeforucs(start, None, None, 0)
    # 按照路径消耗进行排序的FIFO,低路径消耗在前面
    _frontier_priority = []
    frontier_priority_add(node)
    explored = []
    
    while True:
        if len(_frontier_priority) == 0: #means no start 
            return False
        node = _frontier_priority.pop(0)
        if node.state == end:
            return node
        explored.append(node.state) #start is end
        
        for i in range(len())

    states = PriorityQueue()
    states.push((start, []), 0)
    while not states.isEmpty():
        state, actions = states.pop()
        if state == end:
            return actions
        if state not in exploredState:
            successors = end
            for succ in successors:
                coordinates = succ
                if coordinates not in exploredState:
                    directions = succ
                    newCost = actions + [directions]
                    states.push(
                        (coordinates, actions + [directions]), maze.getCostOfActions(newCost))
        exploredState.append(state)
    return actions
    util.raiseNotDefined()
    # return usc_calculator(maze, start, end)

    # return []
    # 
    # 
    # class Node():
    """A node class for A* Path finding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position



        def frontier_priority_add(node):
    """
    :param Node node:
    :return:
    """
    global _frontier_priority
    size = len(_frontier_priority)
    for i in range(size):
        if node.path_cost < _frontier_priority[i].path_cost:
            _frontier_priority.insert(i, node)
            return
    _frontier_priority.append(node)



class Nodeforucs:
    """A node class for UCS Path finding"""

    def __init__(self, state, parent, action, path_cost):
        self.state = state  # now
        self.parent = parent  # parent
        self.action = action  # path
        self.path_cost = path_cost  # cost
    # 
    # 
    # '''
