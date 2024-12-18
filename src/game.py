import pygame
import sys
import numpy as np
from .constants import *
from .entities.snake import Snake
from .entities.food import Food
from .entities.obstacle import Obstacle
from .ui.menu import Menu
from .ui.grid import draw_grid
from .algorithms import astar, bfs, backtracking, simulated_annealing
import matplotlib.pyplot as plt
from collections import defaultdict
from .q_learning.qlearning import QLearning
from .algorithms.node import Node
import csv
import os

class Game:
    def __init__(self, display_game=True):
        pygame.init()
        pygame.display.set_caption("Cá chép hoá rồng!!")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.display_game = display_game
        self.currPath = None  # Thêm biến để lưu đường đi hiện tại
        self.pathIndex = 0      # Thêm biến để theo dõi vị trí hiện tại trong đường đi

        self.reset_game()

    # Hàm dùng để reset game
    def reset_game(self):
        self.grid = init_grid()
        self.snake = Snake()
        self.obstacles = Obstacle()
        self.food = Food()
        self.food.randomize_position(
            self.grid, self.snake.positions, self.obstacles.positions
        )
        self.score = 0

    # Hàm để khởi động game, game có 2 chế độ là thuật toán và trai, ai là chế độ train
    def run(self):
        while True:
            algorithm = Menu.show_main_menu(self.screen)
            if algorithm == "COMPARE":
                print("So sánh hiệu suất...")
                self.compare_algorithms()
                continue
            self.reset_game()
            running = True
            while running:
                self.clock.tick(FPS)
                if not self.handle_events():
                    break
                if not self.update(algorithm):
                    self.surface.fill(BLACK)
                    game_over_font = pygame.font.SysFont("monospace", 40, bold=True)  # Thay đổi kích thước font
                    game_over_text = game_over_font.render("Game Over!!", True, (255, 0, 0))
                    self.screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 20))
                    c_font = pygame.font.SysFont("monospace", 30, bold=True)
                    c_text = c_font.render("Press Space to Continue", True, (255, 255, 255))
                    self.screen.blit(c_text, (SCREEN_WIDTH // 2 - c_text.get_width() // 2, SCREEN_HEIGHT // 2 + 20))
                    pygame.display.update()
                    space = True
                    while space:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    space = False
                    break
                self.draw()

    # Hàm dùng để vẽ có object
    def draw(self):
        self.surface.fill(BLACK)
        draw_grid(self.surface)
        self.obstacles.draw(self.surface)
        self.snake.draw(self.surface)
        self.food.draw(self.surface)
        self.screen.blit(self.surface, (0, 0))
        
        # Vẽ score
        score_text = self.font.render(f"Score {self.score}", True, (235, 91, 0))
        self.screen.blit(score_text, (5, 10))
        
        # Màu cầu vồng
        rainbow_colors = [
            (255, 0, 0),    
            (255, 127, 0),  
            (255, 255, 0),  
            (0, 255, 0),    
            (0, 0, 255),    
            (75, 0, 130),   
            (148, 0, 211)   
        ]
        
        color_index = (pygame.time.get_ticks() // 1000) % len(rainbow_colors)
        current_color = rainbow_colors[color_index]
        name1_text = self.font.render("22110184 Lê Quốc Nam", True, current_color)
        name2_text = self.font.render("22110187 Lê Chí Nghĩa", True, current_color)
        self.screen.blit(name1_text, (5, 40)) 
        self.screen.blit(name2_text, (5, 70))  
        
        pygame.display.update()

    # Hàm xử lí out game và pause game
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return self.pause_game()
        return True

    # Hàm dùng để dừng game tạm thời
    def pause_game(self):
        btn_continue, btn_restart = Menu.show_pause_menu(self.screen)
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return True

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_continue.collidepoint(event.pos):
                        return True
                    elif btn_restart.collidepoint(event.pos):
                        self.screen.fill(BLACK)
                        pygame.display.update()
                        return False
        return True

    def update(self, algorithm=None):
        if algorithm == "Q-learning":
            state_size = (
                GRID_WIDTH * GRID_HEIGHT
            )  
            action_size = 4  

            q_table = np.load("q_table.npy")
            agent = QLearning(state_size, action_size)
            agent.q_table = q_table
            state = agent.get_state(self)
            action = agent.get_action(state)
            direction = [UP, DOWN, LEFT, RIGHT][action]
            self.snake.turn(direction)
        else:
            if self.currPath is None or self.pathIndex >= len(self.currPath) - 1:
                start_pos = (
                    self.snake.get_head_position()[0] / GRIDSIZE,
                    self.snake.get_head_position()[1] / GRIDSIZE,
                )
                goal_pos = (
                    self.food.position[0] / GRIDSIZE,
                    self.food.position[1] / GRIDSIZE,
                )

                # Tìm đường đi theo thuật toán được chọn
                if algorithm == "A*":
                    self.currPath = astar.a_star(
                        start_pos, goal_pos, self.grid, self.obstacles.positions
                    )
                elif algorithm == "BFS":
                    self.currPath = bfs.bfs(
                        start_pos, goal_pos, self.grid, self.obstacles.positions
                    )
                elif algorithm == "BACKTRACKING":
                    self.currPath = backtracking.backtracking(
                        start_pos, goal_pos, self.grid, self.obstacles.positions
                    )
                elif algorithm == "SA":
                    self.currPath = simulated_annealing.simulated_annealing(
                        start_pos, goal_pos, self.grid, self.obstacles.positions
                    )
                
                self.pathIndex = 0
                
                if self.currPath is None or len(self.currPath) < 2:
                    return False

            # Đảm bảo pathIndex không vượt quá độ dài của currPath
            if self.pathIndex < len(self.currPath) - 1:
                current_pos = (
                    self.snake.get_head_position()[0] / GRIDSIZE,
                    self.snake.get_head_position()[1] / GRIDSIZE,
                )
                next_pos = self.currPath[self.pathIndex + 1]
                direction = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
                
                self.snake.turn(direction)
                self.pathIndex += 1

        if not self.snake.move(self.grid, self.obstacles.positions):
            return False

        if self.snake.get_head_position() == self.food.position:
            self.snake.length += 1
            self.score = self.snake.length - 1
            self.food.randomize_position(
                self.grid, self.snake.positions, self.obstacles.positions
            )
            self.currPath = None  # Reset đường đi khi ăn được thức ăn
            
        return True
    
    # Hàm để so sánh các chỉ số
    def compare_algorithms(self):
        algorithms = ["BFS", "A*", "BACKTRACKING", "SA"]
        num_of_algo = NUM_COMPARE
        stats = defaultdict(lambda: defaultdict(list))

        for i in algorithms:
            for _ in range(num_of_algo):
                self.reset_game()
                score, moves, time_taken = self.run_algorithm(i)
                score = max(1, score)  
                stats[i]["scores"].append(score)
                stats[i]["moves"].append(moves)  
                stats[i]["time"].append(time_taken)  

        # Tạo thư mục 'reports' trong 'src' nếu chưa tồn tại
        if not os.path.exists('src/reports'):
            os.makedirs('src/reports')

        # Ghi dữ liệu vào file CSV trong thư mục 'src/reports'
        with open('src/reports/report.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Algorithm", "Run", "Score", "Moves", "Time"])
            for algo in algorithms:
                for run, (score, moves, time) in enumerate(zip(stats[algo]["scores"], stats[algo]["moves"], stats[algo]["time"]), start=1):
                    writer.writerow([algo, run, score, moves, time])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Hiệu suất các thuật toán")

        metrics = {
            "scores": ("Điểm số", ax1),
            "moves": ("Số bước(1 gridsize)", ax2),
            "time": ("Thời gian(giây)", ax3),
        }

        for metric, (ylabel, ax) in metrics.items():
            for algo in algorithms:
                values = stats[algo][metric]
                runs = range(1, len(values) + 1)
                line = ax.plot(runs, values, marker="o", label=algo)[0]
                
                for x, y in zip(runs, values):
                    if metric in ["time", "moves"]:
                        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                                  xytext=(0,10), ha='center')
                    else:
                        ax.annotate(f'{int(y)}', (x, y), textcoords="offset points", 
                                  xytext=(0,10), ha='center')

            ax.set_xlabel("Lần chạy")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    # Hàm dùng để chạy thuật toán và đánh giá
    def run_algorithm(self, algo):
        import time
        start_time = time.time()
        moves = 0
        self.reset_game()
        running = True
        while running:  
            if not self.update(algorithm=algo):
                break
            moves += 1
            if self.display_game:
                self.draw()
                self.clock.tick(FPS)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
        time_taken = time.time() - start_time
        return self.score, moves, time_taken
