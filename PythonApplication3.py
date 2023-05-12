import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from queue import PriorityQueue

# 定義迷宮大小
maze_size = (90, 90)

# 隨機生成迷宮
maze = np.random.choice([0, 1], size=maze_size, p=[0.7, 0.3])

# 定義起點和終點
start = (np.random.randint(0, maze_size[0]), np.random.randint(0, maze_size[1]))
while maze[start] == 1:
    start = (np.random.randint(0, maze_size[0]), np.random.randint(0, maze_size[1]))

end = (np.random.randint(0, maze_size[0]), np.random.randint(0, maze_size[1]))
while maze[end] == 1:
    end = (np.random.randint(0, maze_size[0]), np.random.randint(0, maze_size[1]))

# 定義可能的移動方向
directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # 上、左、下、右

# 初始化走迷宮的過程紀錄
maze_process = []

# 初始化動畫的圖片
fig, ax = plt.subplots()
ax.set_title('Maze Solving Animation')
ax.set_xticks([])
ax.set_yticks([])

# 定義更新每一個動畫幀的函式
def update(frame):
    ax.clear()
    ax.set_title(f'Maze Solving Animation (Frame {frame+1})')

    # 繪出整體迷宮
    ax.imshow(maze, cmap='binary')
    ax.scatter(start[1], start[0], c='g', marker='o', label='Start')
    ax.scatter(end[1], end[0], c='r', marker='o', label='End')

    # 標記走過的路線
    for i in range(frame + 1):
        x, y = maze_process[i]
        ax.scatter(y, x, c='b', marker='o', label='Process' if i == 0 else '')

    # 標記最短路徑
    if frame == len(shortest_path) - 1:
        for node in shortest_path:
            x, y = node
            ax.scatter(y, x, c='m', marker='o', label='Shortest Path' if node == shortest_path[-1] else '')

    ax.legend()

# 定義Dijkstra演算法
def dijkstra(start, end):
    # 初始化距離和前一個節點
    distance = np.full(maze_size, np.inf)
    distance[start] = 0
    prev = np.zeros(maze_size, dtype=np.ndarray)
    prev[start] = None

    # 初始化PriorityQueue來進行節點擴展
    pq = PriorityQueue()
    pq.put((0, start))

    while not pq.empty():
        _, current = pq.get()

        if current == end:
            break

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < maze_size[0] and 0 <=            ny < maze_size[1] and maze[nx][ny] != 1:  # 確保新節點在迷宮內且不是障礙物
                new_cost = distance[current] + 1  # 新節點的距離是當前節點距離加1

                if new_cost < distance[nx][ny]:
                    distance[nx][ny] = new_cost
                    prev[nx][ny] = current
                    pq.put((new_cost, (nx, ny)))

    # 若終點未被擴展到，表示無法到達終點，返回空的最短路徑
    if prev[end] is None:
        return []

    # 從終點回溯找到最短路徑
    shortest_path = [end]
    while prev[shortest_path[-1]] is not None:
        shortest_path.append(prev[shortest_path[-1]])
    shortest_path.reverse()

    return shortest_path

# 使用Dijkstra演算法找到最短路徑
shortest_path = dijkstra(start, end)

# 將最短路徑添加到走迷宮的過程紀錄中
maze_process += shortest_path

# 將最短路徑從坐標轉換為整數索引
shortest_path_indices = [(p[0] * maze_size[1] + p[1]) for p in shortest_path]

# 初始化動畫
ani = animation.FuncAnimation(fig, update, frames=len(maze_process), interval=1, blit=False)

# 顯示動畫
plt.show()


