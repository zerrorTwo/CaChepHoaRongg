import numpy as np
from ..constants import GRIDSIZE, GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT
import glob
import re
import os

# 16 (nguy hiểm) * 3 (hướng thức ăn x ) * 3 (hướng thức ăn y) = 144


class QLearning:
    def __init__(self, size_of_state, size_of_action, with_obstacles=False):
        self.size_of_state = size_of_state
        self.size_of_action = size_of_action
        self.learning_rate = 0.1  # tỷ lệ học
        self.discount_factor = 0.95  # hệ số giảm 
        self.epsilon = 0.1  # tỷ lệ khám phá ngẫu nhiên
        self.with_obstacles = with_obstacles
        # Tạo thư mục models nếu chưa tồn tại
        if with_obstacles:
            self.model_dir = os.path.join(os.path.dirname(__file__), "modelsObstacle")
        else:
            self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Tìm file q_table có điểm số cao nhất
        files = glob.glob(os.path.join(self.model_dir, "q_table_*.npy"))
        best_score = -1
        best_table = None

        for file in files:
            score = int(re.findall(r"q_table_(\d+).npy", os.path.basename(file))[0])
            if score > best_score:
                best_score = score
                best_table = file

        if best_table:
            self.q_table = np.load(best_table)
            print(f"Đã tải Q-table từ file {os.path.basename(best_table)}")
        else:
            print("Tạo Q-table mới")
            self.q_table = np.zeros((size_of_state, size_of_action))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.size_of_action)
        else:
            return np.argmax(self.q_table[state])

    # cập nhật Q-table với số điểm thưởng
    def update(self, state, action, reward, next_state, done):
        if done:
            point = reward
        else:
            point = reward + self.discount_factor * np.amax(self.q_table[next_state])

        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][
            action
        ] + self.learning_rate * point

        # giảm tỷ lệ khám phá ngẫu nhiên sau mỗi lần
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def get_state(self, game):
        head_x, head_y = game.snake.get_head_position()
        food_x, food_y = game.food.position

        head_x = int(head_x / GRIDSIZE)
        head_y = int(head_y / GRIDSIZE)
        food_x = int(food_x / GRIDSIZE)
        food_y = int(food_y / GRIDSIZE)

        # Tính toán hướng tương đối của thức ăn
        food_dir_x = 0
        food_dir_y = 0
        if food_x < head_x:
            food_dir_x = -1
        elif food_x > head_x:
            food_dir_x = 1
        if food_y < head_y:
            food_dir_y = -1
        elif food_y > head_y:
            food_dir_y = 1

        # Kiểm tra nguy hiểm ở 4 hướng
        array_consider = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for i, (dx, dy) in enumerate(directions):
            next_x = head_x + dx
            next_y = head_y + dy

            # Kiểm tra va chạm với tường
            if (
                next_x < 0
                or next_x >= GRID_WIDTH
                or next_y < 0
                or next_y >= GRID_HEIGHT
            ):
                array_consider[i] = 1
            elif (next_x * GRIDSIZE, next_y * GRIDSIZE) in game.obstacles.positions:
                array_consider[i] = 1
            elif (next_x * GRIDSIZE, next_y * GRIDSIZE) in game.snake.positions[1:]:
                array_consider[i] = 1

        # Thêm thông tin về chướng ngại vật vào trạng thái
        obstacle_state = sum(
            2**i for i, pos in enumerate(game.obstacles.positions)
            if (head_x, head_y) == (pos[0] // GRIDSIZE, pos[1] // GRIDSIZE)
        )

        if self.with_obstacles:

            state = (
                array_consider[0] * 1
                + array_consider[1] * 2
                + array_consider[2] * 4
                + array_consider[3] * 8
                + (food_dir_x + 1) * 16
                + (food_dir_y + 1) * 48
                + obstacle_state * 144  # Thêm thông tin chướng ngại vật
            )
        else:
            state = (
                array_consider[0] * 1
                + array_consider[1] * 2
                + array_consider[2] * 4
                + array_consider[3] * 8
                + (food_dir_x + 1) * 16
                + (food_dir_y + 1) * 48
            )

        return state
