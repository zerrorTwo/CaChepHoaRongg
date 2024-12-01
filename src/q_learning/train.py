# run in terminal python -m src.q_learning.train to train
# python -m src.q_learning.train --with_obstacles to train with obstacles
from src.game import Game
from src.q_learning.qlearning import QLearning
from ..constants import *
import numpy as np
import pygame
import os
import time
import argparse

def train(with_obstacles):
    # Tạo đường dẫn đến thư mục models
    if with_obstacles:
        model_dir = os.path.join(os.path.dirname(__file__), "modelsObstacle")
    else:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    # Tải điểm cao nhất từ file nếu tồn tại
    best_path = os.path.join(model_dir, "best_score.npy")
    try:
        best_data = np.load(best_path)
        best_score = int(best_data)
    except:
        best_score = 0

    game = Game()
    size_of_state = 144  # 2 * 2 * 2 * 2 (danger) * 3 (food_dir_x) * 3 (food_dir_y)
    size_of_action = 4  # UP, DOWN, LEFT, RIGHT
    agent = QLearning(size_of_state, size_of_action, with_obstacles)

    num_of_train = 1000000
    max_steps = 100000

    # Đặt vị trí cố định cho chướng ngại vật nếu cần
    if with_obstacles:
        fixed_obstacles = [(5, 5), (10, 10), (15, 15), (3, 7), (7, 3), (12, 8), (8, 12),
                          (2, 14), (14, 2), (6, 9), (9, 6), (11, 4), (4, 11), (13, 7),
                          (7, 13), (3, 15), (15, 3), (8, 8), (5, 12), (12, 5), (2, 9),
                          (9, 2), (6, 14), (14, 6), (4, 7), (7, 4), (11, 10), (10, 11),
                          (13, 3), (3, 13), (8, 5), (5, 8), (12, 12), (2, 6), (6, 2),
                          (9, 15), (15, 9), (4, 4), (10, 7), (7, 10)]  # 40 vị trí cố định cho vật cản
        game.obstacles.positions = [(x * GRIDSIZE, y * GRIDSIZE) for x, y in fixed_obstacles]

    for episode in range(num_of_train):
        game.reset_game()
        if with_obstacles:
            game.obstacles.positions = [(x * GRIDSIZE, y * GRIDSIZE) for x, y in fixed_obstacles]
        state = agent.get_state(game)
        total_reward = 0

        for step in range(max_steps):
            # game.draw()
            # pygame.display.flip()
            # time.sleep(0.05)

            action = agent.get_action(state)

            # Chuyển đổi action thành hướng di chuyển
            direction = [UP, DOWN, LEFT, RIGHT][action]
            game.snake.turn(direction)

            if not game.snake.move(game.grid, game.obstacles.positions):
                reward = -100
                done = True
            elif game.snake.get_head_position() == game.food.position:
                reward = 10
                game.snake.length += 1
                game.score = game.snake.length - 1
                game.food.randomize_position(
                    game.grid, game.snake.positions, game.obstacles.positions
                )
                done = False
            else:
                current_dis = (
                    (state % (GRID_WIDTH * GRID_HEIGHT)) // GRID_WIDTH
                    - game.food.position[0] // GRIDSIZE
                ) ** 2 + (
                    (state % (GRID_WIDTH * GRID_HEIGHT)) % GRID_WIDTH
                    - game.food.position[1] // GRIDSIZE
                ) ** 2
                next_dis = (
                    game.snake.get_head_position()[0] // GRIDSIZE
                    - game.food.position[0] // GRIDSIZE
                ) ** 2 + (
                    game.snake.get_head_position()[1] // GRIDSIZE
                    - game.food.position[1] // GRIDSIZE
                ) ** 2
                reward = 1 if next_dis < current_dis else -1
                done = False

            next_state = agent.get_state(game)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break
        if episode % 1 == 0:
            print(
                f"Episode: {episode}, Score: {game.score}, Total Reward: {total_reward}"
            )

        # Chỉ lưu khi đạt điểm số cao hơn điểm cao nhất mọi thời đại
        if game.score > best_score:
            best_score = game.score
            # Lưu q_table với tên file chứa điểm số
            q_table_path = os.path.join(model_dir, f"q_table_{best_score}.npy")
            np.save(q_table_path, agent.q_table)
            np.save(best_path, np.array(best_score))
            print(f"Đã lưu Q-table mới với điểm số cao nhất: {best_score}")

if __name__ == "__main__":
    # Sử dụng argparse để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Train the Q-learning model.")
    parser.add_argument('--with_obstacles', action='store_true', help='Huấn luyện với chướng ngại vật')
    args = parser.parse_args()

    train(args.with_obstacles)
