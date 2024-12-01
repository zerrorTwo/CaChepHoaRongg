import heapq 
from .node import Node
from ..constants import *

# Hàm tính khoảng cách bằng Manhattan
def heuristic(node, goal_node):
    return abs(node.position[0] - goal_node.position[0]) + \
        abs(node.position[1] - goal_node.position[1])

def a_star(start_pos, goal_pos, grid, obstacles):
    open_list = [] # danh sách nút có thể đi
    visited = set() 

    start_node = Node(start_pos)
    goal_node = Node(goal_pos)

    # Dùng heapq để có thể lấy các đường có cost nhỏ nhất
    heapq.heappush(open_list, (0 + heuristic(start_node, goal_node), start_node))

    dic_gcosts = {start_node: 0}
    dic_hcosts = {}

    while open_list:
        _, current_node = heapq.heappop(open_list)
        visited.add(current_node)

        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.append(start_node.position)
            return path[::-1]

        for child in current_node.get_neighbors(grid, obstacles):
            if child in visited:
                continue

            g_current = dic_gcosts[current_node] + 1
            h_current = heuristic(child, goal_node)
            f_current = g_current + h_current

            if child not in dic_gcosts or g_current < dic_gcosts[child]:
                dic_gcosts[child] = g_current
                dic_hcosts[child] = h_current
                heapq.heappush(open_list, (f_current, child))

    return None