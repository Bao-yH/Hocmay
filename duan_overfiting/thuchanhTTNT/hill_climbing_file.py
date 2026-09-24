import time

# === Hàm đọc file đồ thị ===
def read_graph(file_path):
    graph = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            u, v, cost = parts
            cost = float(cost)
            if u not in graph:
                graph[u] = {}
            graph[u][v] = cost
            if v not in graph:
                graph[v] = {}
    return graph

# === Hàm đọc file heuristic ===
def read_heuristic(file_path):
    h = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            node, value = parts
            h[node] = float(value)
    return h

# === Thuật toán leo đồi ===
def hill_climbing(graph, heuristic, start, goal):
    current = start
    path = [current]
    total_cost = 0
    steps = []
    start_time = time.time()

    while current != goal:
        neighbors = graph[current]
        if not neighbors:
            break

        best_neighbor = None
        best_h = heuristic[current]
        best_cost = 0

        for n, c in neighbors.items():
            h_val = heuristic.get(n, float('inf'))
            steps.append((current, n, h_val))
            if h_val < best_h:
                best_h = h_val
                best_neighbor = n
                best_cost = c

        if best_neighbor is None:
            break

        current = best_neighbor
        path.append(current)
        total_cost += best_cost

    end_time = time.time()
    return path, total_cost, steps, end_time - start_time

# === Chạy chương trình ===
graph = read_graph('CANH.txt')
heuristic = read_heuristic('UOCLUONG.txt')

start_node = 'A'
goal_node = 'B'

path, cost, steps, runtime = hill_climbing(graph, heuristic, start_node, goal_node)

print("=== KẾT QUẢ THUẬT TOÁN LEO ĐỒI ===")
print(f"Đường đi: {' -> '.join(path)}")
if path[-1] == goal_node:
    print(f" Tìm thấy đích {goal_node}")
else:
    print(f" Dừng tại đỉnh {path[-1]} (local optimum)")
print(f"Tổng chi phí đường đi: {cost}")
print(f"Thời gian chạy: {runtime:.6f} giây")

print("\n=== Quá trình duyệt ===")
for step in steps:
    print(f"Từ {step[0]} xét {step[1]} có h = {step[2]}")
