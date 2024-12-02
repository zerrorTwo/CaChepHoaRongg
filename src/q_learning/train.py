# run in terminal python -m src.q_learning.train to train
# Yêu cầu chỉnh constants cho GRIDSIZE bằng 40 
from src.game import Game
from src.q_learning.qlearning import QLearning
from ..constants import *
import numpy as np
import pygame
import os
import time
from src.constants import FPS

def train():
    # Tạo đường dẫn đến thư mục models
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    # Tải điểm cao nhất từ file nếu tồn tại
    best_path = os.path.join(model_dir, "best_score.npy")
    try:
        best_data = np.load(best_path)
        best_score = int(best_data)
    except:
        best_score = 0

    pos_obs =[(0, 0), (3, 5), (6, 10), (9, 15), (12, 2), (15, 7), (0, 12), (3, 17), (6, 4), (9, 9),
                      (12, 14), (15, 1), (0, 6), (3, 11), (6, 16), (9, 3), (12, 8), (15, 13), (0, 18), (3, 5),
                      (7, 11), (11, 17), (15, 4), (2, 10), (6, 16), (10, 3), (14, 9), (1, 15), (5, 2), (9, 8),
                      (13, 14), (0, 1), (4, 7), (8, 13), (12, 0), (16, 6), (3, 12), (7, 18), (11, 5), (15, 11)]

    game = Game()
    pos_obs = pos_obs
    game.obstacles.positions = [(x * GRIDSIZE, y * GRIDSIZE) for x, y in pos_obs]
    size_of_state = 144  
    size_of_action = 4  
    agent = QLearning(size_of_state, size_of_action)

    num_of_train = 1000000
    max_steps = 1000000

    clock = pygame.time.Clock() 

    for number in range(num_of_train):
        game.reset_game()
        game.obstacles.positions = [(x * GRIDSIZE, y * GRIDSIZE) for x, y in pos_obs]
        game.food.randomize_position(game.grid, game.snake.positions, game.obstacles.positions)
        state = agent.get_state(game)
        sum_reward = 0

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            game.draw()
            pygame.display.flip()
            clock.tick(FPS)

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
            sum_reward += reward

            if done:
                break
        if number % 1 == 0:
            print(
                f"Lần: {number}, Điểm: {game.score}, Phần thưởng: {sum_reward}"
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
    train()