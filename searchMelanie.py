import argparse
import heapq
#Melanie B 6261665
#AI HW #2


def load_maze(filename):
    with open(filename, 'r') as f:
        maze = [list(line.strip()) for line in f.readlines()]
    start, goal = None, None
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 'S':
                start = (i, j)
            elif maze[i][j] == 'G':
                goal = (i, j)
    return maze, start, goal

def get_next_moves(position, maze):
        x, y = position
        moves = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        next_moves = [move for move in moves if 0 <= move[0] < len(maze) 
                       and 0 <= move[1] < len(maze[0]) 
                       and maze[move[0]][move[1]] != '%']
        return next_moves

# Example: depth-first search (DFS)
# Note: DFS may not find the optimal path
def depth_first_search(maze, start, goal):
    """
    Perform depth-first search on the maze.
    - DFS explores as far as possible along each branch before backtracking.
    - It may not find the optimal path and can get stuck in infinite loops.
    """
    
    stack = [(start, [start]), ]
    visited = set()
    while stack:
        current, path = stack.pop()
        if current == goal:
            return path
        if current not in visited:
            visited.add(current)
            for move in get_next_moves(current, maze):
                if move not in visited:
                    new_path = list(path)
                    new_path.append(move)
                    stack.append((move, new_path))
    return []

def astar_search(maze, start, goal):
    """
        Perform A* search on the maze.
        - A* uses a heuristic function to estimate the cost from the current position to the goal.
        - It considers both the cost to reach the current position and the estimated cost to reach the goal.
        - A* aims to find the optimal path while considering the estimated cost.
        """
    def calcHeuristicVal(position, goal):
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    priority_queue = [(calcHeuristicVal(start, goal), start, [start])]
    visited = set()

    while priority_queue:
        _, current, path = heapq.heappop(priority_queue)

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)

            for move in get_next_moves(current, maze):
                if move not in visited:
                    new_path = list(path)
                    new_path.append(move)
                    heapq.heappush(priority_queue, (len(new_path) + calcHeuristicVal(move, goal), move, new_path))

    return []


def breadth_first_search(maze, start, goal):
    """
    Perform breadth-first search on the maze.
    - BFS explores all paths at the current depth before moving on to the next depth.
    - It is guaranteed to find the shortest path.
    """
    queue = [(start, [start])]
    visited = set()

    while queue:
        current, path = queue.pop(0)

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)

            for move in get_next_moves(current, maze):
                if move not in visited:
                    new_path = list(path)
                    new_path.append(move)
                    queue.append((move, new_path))

    return []

# This is part of the tests:
def test_start_goal(path, start, goal):
    assert(path[0] == start)
    assert(path[-1] == goal)


def test_path(path, maze):
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    for i, p in enumerate(path):
        assert(maze[p[0]][p[1]] != "%")
        if i > 0:
            assert(manhattan_distance(path[i], path[i-1]) == 1)

def main():
    parser = argparse.ArgumentParser(description='Search Algorithms for Solving Maze Problems')
    parser.add_argument('--method', required=True, choices=["depth-first", "breadth-first", "astar"],
                        help='Search method to use')
    parser.add_argument('--maze', type=str, help='Path to the maze file')
    args = parser.parse_args()
    maze, start, goal = load_maze(args.maze) 
    if args.method == "depth-first":
        path = depth_first_search(maze, start, goal)
    elif args.method == "breadth-first":
        path = breadth_first_search(maze, start, goal)
    elif args.method == "astar":
        path = astar_search(maze, start, goal)

    # this is simple tests:
    test_start_goal(path, start, goal)
    test_path(path, maze)

    for p in path:
        if maze[p[0]][p[1]] not in ['S', 'G']:
            maze[p[0]][p[1]] = 'x'
    print("\n".join(["".join(row) for row in maze]))
    print("\nPath Cost:", len(path))

if __name__ == "__main__":
    main()

#Outputs Explanations:

# Depth-First Search (DFS):
# DFS may not find the optimal path.
# The path might take longer routes, and the cost may not be minimized.
#
# Breadth-First Search (BFS):
# BFS explores all paths at the current depth before moving on to the next depth.
# It is guaranteed to find the shortest path.
#
# A Search (Astar):*
# A* uses a heuristic function to estimate the cost from the current position to the goal.
# It considers both the cost to reach the current position and the estimated cost to reach the goal.
# A* aims to