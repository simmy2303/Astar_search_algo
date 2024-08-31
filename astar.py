import networkx as nx
import heapq

# Define the graph
graph = nx.Graph()

# Add nodes
graph.add_node('1', pos=(0, 0))
graph.add_node('2', pos=(0, 1))
graph.add_node('3', pos=(0, 2))
graph.add_node('5', pos=(0, 4))
graph.add_node('6', pos=(0, 5))

graph.add_node('7', pos=(1, 0))
graph.add_node('8', pos=(1, 1), label='T2')
graph.add_node('9', pos=(1, 2))
graph.add_node('10', pos=(1, 3), label='R1')
graph.add_node('11', pos=(1, 4))
graph.add_node('12', pos=(1, 5))

graph.add_node('13', pos=(2, 0))
graph.add_node('14', pos=(2, 1))
graph.add_node('16', pos=(2, 3))
graph.add_node('17', pos=(2, 4), label='T2')
graph.add_node('18', pos=(2, 5))

graph.add_node('19', pos=(3, 0))
graph.add_node('20', pos=(3, 1), label='T4')
graph.add_node('21', pos=(3, 2))
graph.add_node('23', pos=(3, 4), label='Treasure')
graph.add_node('24', pos=(3, 5))

graph.add_node('25', pos=(4, 0), label='R1')
graph.add_node('26', pos=(4, 1), label='Treasure')
graph.add_node('28', pos=(4, 3))
graph.add_node('30', pos=(4, 5))

graph.add_node('31', pos=(5, 0))
graph.add_node('32', pos=(5, 1))
graph.add_node('33', pos=(5, 2))
graph.add_node('34', pos=(5, 3), label='T3')
graph.add_node('35', pos=(5, 4))
graph.add_node('36', pos=(5, 5), label='R2')

graph.add_node('37', pos=(6, 0))
graph.add_node('38', pos=(6, 1), label='T3')
graph.add_node('39', pos=(6, 2))
graph.add_node('42', pos=(6, 5))

graph.add_node('43', pos=(7, 0))
graph.add_node('44', pos=(7, 1))
graph.add_node('45', pos=(7, 2), label='R2')
graph.add_node('46', pos=(7, 3), label='Treasure')
graph.add_node('48', pos=(7, 5))

graph.add_node('49', pos=(8, 0))
graph.add_node('51', pos=(8, 2), label='T1')
graph.add_node('52', pos=(8, 3))
graph.add_node('53', pos=(8, 4))
graph.add_node('54', pos=(8, 5))

graph.add_node('55', pos=(9, 0))
graph.add_node('56', pos=(9, 1))
graph.add_node('57', pos=(9, 2))
graph.add_node('58', pos=(9, 3), label='Treasure')
graph.add_node('59', pos=(9, 4))
graph.add_node('60', pos=(9, 5))

# Neighbours of each node
edges = [
    ('1', '2'), ('1', '7'), ('1', '8'), 
    ('2', '3'), ('2', '8'), ('2', '9'), 
    ('3', '9'), ('3', '10'), 
    ('5', '6'), ('5', '11'), 
    ('6', '12'),
    ('7', '1'), ('7', '8'), ('7', '13'), 
    ('8', '1'), ('8', '2'), ('8', '7'), ('8', '9'), ('8', '13'), ('8', '14'), 
    ('9', '2'), ('9', '3'), ('9', '8'), ('9', '10'), ('9', '14'), 
    ('10', '3'), ('10', '9'), ('10', '11'),('10', '16'),
    ('11', '5'), ('11', '10'), ('11', '12'), ('11', '16'), ('11', '17'), 
    ('12', '5'), ('12', '6'), ('12', '11'), ('12', '17'), ('12', '18'),
    ('13', '7'), ('13', '8'), ('13', '14'), ('13', '19'), ('13', '20'), 
    ('14', '8'), ('14', '9'), ('14', '13'), ('14', '20'), ('14', '21'), 
    ('16', '10'), ('16', '11'), ('16', '17'), ('16', '23'), 
    ('17', '11'), ('17', '12'), ('17', '16'), ('17', '18'), ('17', '23'), ('17', '24'), 
    ('18', '12'), ('18', '17'), ('18', '24'), 
    ('19', '13'), ('19', '20'), ('19', '25'),
    ('20', '13'), ('20', '14'), ('20', '19'), ('20', '21'), ('20', '25'), ('20', '26'), 
    ('21', '14'), ('21', '20'), ('21', '26'), 
    ('23', '16'), ('23', '17'), ('23', '24'), ('23', '28'), 
    ('24', '17'), ('24', '18'), ('24', '23'), ('24', '30'), 
    ('25', '19'), ('25', '26'), ('25', '31'), ('25', '32'), 
    ('26', '20'), ('26', '21'), ('26', '25'), ('26', '32'), ('26', '33'), 
    ('28', '23'), ('28', '34'), ('28', '35'),
    ('30', '24'), ('30', '36'),
    ('31', '25'), ('31', '32'), ('31', '37'), 
    ('32', '25'), ('32', '26'), ('32', '31'), ('32', '33'), ('32', '37'), ('32', '38'), 
    ('33', '26'), ('33', '32'), ('33', '34'), ('33', '38'), ('33', '39'), 
    ('34', '28'), ('34', '33'), ('34', '35'), ('34', '39'), 
    ('35', '28'), ('35', '34'), ('35', '36'), 
    ('36', '30'), ('36', '42'), 
    ('37', '31'), ('37', '32'), ('37', '38'), ('37', '43'), ('37', '44'), 
    ('38', '32'), ('38', '33'), ('38', '37'), ('38', '39'), ('38', '44'), ('38', '45'), 
    ('39', '33'), ('39', '34'), ('39', '38'), ('39', '45'), ('39', '46'), 
    ('42', '36'), ('42', '48'),  
    ('43', '37'), ('43', '44'), ('43', '49'), 
    ('44', '37'), ('44', '38'), ('44', '43'), ('44', '45'),  ('44', '49'), 
    ('45', '38'), ('45', '39'), ('45', '44'), ('45', '46'), ('45', '51'), 
    ('46', '39'), ('46', '45'), ('46', '51'), ('46', '52'), 
    ('48', '42'), ('48', '53'), ('48', '54'), 
    ('49', '43'), ('49', '44'), ('49', '55'), ('49', '56'), 
    ('51', '45'), ('51', '46'), ('51', '52'), ('51', '57'), ('51', '58'), 
    ('52', '46'), ('52', '51'), ('52', '53'), ('52', '58'), ('52', '59'), 
    ('53', '48'), ('53', '52'), ('53', '54'), ('53', '59'), ('53', '60'), 
    ('55', '49'), ('55', '56'), 
    ('56', '49'), ('56', '57'), ('56', '58'), 
    ('57', '51'), ('57', '58'), 
    ('58', '51'), ('58', '52'), ('58', '57'), ('58', '59'),  
    ('59', '52'), ('59', '53'), ('59', '58'), ('59', '60'), 
    ('60', '53'), ('60', '54'), ('60', '59'), 
]

# Add edges to the graph with a default stepcost of 1
for edge in edges:
    graph.add_edge(*edge, stepcost=1)

# Define the heuristic function (Manhattan distance)
def heuristic(a, b):
    (x1, y1) = graph.nodes[a]['pos']
    (x2, y2) = graph.nodes[b]['pos']
    return abs(x1 - x2) + abs(y1 - y2)

# A* search algorithm
def a_star_search(graph, start, goal, avoid_traps={'T1', 'T2', 'T3', 'T4'}):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in graph.neighbors(current):
            if graph.nodes[neighbor].get('label') in avoid_traps:
                continue

            tentative_g_score = g_score[current] + graph[current][neighbor]['stepcost']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Frontier approach to find the optimal path
def find_optimal_path(graph, start, treasures, avoid_traps={'T1', 'T2', 'T3', 'T4'}):
    frontier = []
    heapq.heappush(frontier, (0, start, [start], set(treasures)))
    min_cost = float('inf')
    best_path = []

    while frontier:
        current_cost, current_position, path, remaining_treasures = heapq.heappop(frontier)

        if not remaining_treasures:
            if current_cost < min_cost:
                min_cost = current_cost
                best_path = path
            continue

        for treasure in remaining_treasures:
            sub_path = a_star_search(graph, current_position, treasure)
            if not sub_path:
                continue

            sub_cost = sum(graph[sub_path[i]][sub_path[i + 1]]['stepcost'] for i in range(len(sub_path) - 1))
            new_cost = current_cost + sub_cost
            new_remaining_treasures = remaining_treasures - {treasure}
            new_path = path + sub_path[1:]

            heapq.heappush(frontier, (new_cost, treasure, new_path, new_remaining_treasures))

    return best_path, min_cost

# Find all treasures
treasures = {node for node, data in graph.nodes(data=True) if data.get('label') == 'Treasure'}

# Collect all treasures starting from node '1'
optimal_path, total_cost = find_optimal_path(graph, '1', treasures)

# Function to print the path details
def display_path_details(graph, path, treasures):
    current_cost = 0
    accumulated_heuristic = 0
    remaining_treasures = set(treasures)
    collected_treasures = []

    # Initial state details
    print("--------------------- Initial State ---------------------")
    print(f"Current Position: {path[0]}")
    print(f"Current Path Cost: {current_cost}")
    print(f"Heuristic Cost: 0")  # Initial heuristic cost is 0
    print(f"Total Path Cost: {current_cost}")

    for i in range(1, len(path)):
        current_position = path[i - 1]
        next_position = path[i]
        step_cost = graph[current_position][next_position]['stepcost']
        current_cost += step_cost
        heuristic_cost = heuristic(next_position, '1') 
        accumulated_heuristic += heuristic_cost
        total_path_cost = current_cost + accumulated_heuristic

        print(f"\n--------------------- Step {i} ---------------------")
        print(f"Current Position: {next_position}")
        print(f"Current Path Cost: {current_cost}")
        print(f"Heuristic Cost: {accumulated_heuristic}")
        print(f"Total Path Cost: {total_path_cost}")

        if next_position in remaining_treasures:
            collected_treasures.append(next_position)
            remaining_treasures.remove(next_position)
            print(f"\n++++++++++++ Treasure {next_position} found! ++++++++++++")
            if remaining_treasures:
                print(f"Remaining Treasures: {remaining_treasures}")
            else:
                print("Final Treasure Found! All Treasures are Collected!")
            print(f"Collected Treasures: {collected_treasures}\n")


# Print the path details for the optimal path
print("====================== Details of the shortest path found to collect all the treasures while avoiding traps ======================\n")
display_path_details(graph, optimal_path, treasures)
print("\n===================== All treasures are collected! Congrats on finding the shortest path to escape the virtual world! ====================\n")

print("Final path to collect all treasures:", optimal_path)

