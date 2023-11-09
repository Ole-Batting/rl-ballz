import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    WIDTH, HEIGHT, P, Q, VEL, RADIUS, SOBELX, SOBELY, SX, SY, BG, X_BALL, Y_BALL,
    K_TILE, SKIP, ANG_STEP, W, H, S_TEXT, T_TEXT, SPLITS
)
from ball import Ball
from tile import Tile


class Board:
    def __init__(self, headless=False, show_every=0):
        self.tiles = []
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.tiles.append(Tile(np.array([j, i]), 0))

        self.ball_init_pos = np.array([K_TILE * 7 // 2, K_TILE * 9 - RADIUS], dtype=float)
        self.balls = [Ball(self.ball_init_pos.copy())]

        self.i = 0
        self.rng = np.random.default_rng()

        frame = np.zeros((HEIGHT * K_TILE, WIDTH * K_TILE), dtype=np.float32)
        cv2.rectangle(frame, (0,0), (WIDTH * K_TILE - 1, HEIGHT * K_TILE + 1), (1), 2)

        normalx = cv2.filter2D(frame, cv2.CV_32F, SOBELX)
        nxn = cv2.dilate(-normalx, SX)
        nxp = cv2.dilate(normalx, SX)
        self.normalx = nxp - nxn

        normaly = cv2.filter2D(frame, cv2.CV_32F, SOBELY)
        nyn = cv2.dilate(-normaly, SY)
        nyp = cv2.dilate(normaly, SY)
        self.normaly = nyp - nyn

        # self.inspect_normals()

        self.any_dead = False
        self.headless = headless
        self.show_every = show_every
        self.show_this = True
        self.resets = 0
        self.video = None
        self.video_frames = []
        self.hits = 0
        self.ball_reward = 0
    
    def get_normal_point(self, x, y):
        return self.normalx[int(y), int(x)], self.normaly[int(y), int(x)]
    
    def inspect_normals(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.normalx)
        ax[1].imshow(self.normaly)
        plt.show()
    
    @property
    def all_balls_dead(self):
        return all([ball.dead for ball in self.balls])
    
    @property
    def waiting_for_shot(self):
        return all([ball.vel == 0 for ball in self.balls])
    
    def next(self):
        self.i += 1
        self.hits = 0
        self.ball_reward = 0
        row = self.rng.random(WIDTH)
        min_used = self.i == 1
        for tile, r in zip(self.tiles[:WIDTH], row):
            if r == min(row) and not min_used:
                tile.c = -1
            elif r > P:
                tile.c = self.i
            elif r < Q:
                tile.c = self.i * 2
            tile.pos[1] = 0
        
        for tile in self.tiles[-WIDTH:]:
            tile.c = 0
        
        for tile in self.tiles:
            tile.roll()
        
        self.tiles = np.roll(self.tiles, WIDTH, axis=0)

        for ball in self.balls:
            ball.vel = 0
            ball.dead = False
            ball.pos = self.ball_init_pos.copy()
        self.any_dead = False

        return any([tile.c > 0 for tile in self.tiles[-WIDTH:]])
    
    def set_dir(self, ang):
        for ball in self.balls:
            ball.dir = np.array([np.cos(ang), -abs(np.sin(ang))])

    def update(self):
        for ball in self.balls:
            ret = ball.update()
            if ret and not self.any_dead:
                self.any_dead = True
                self.ball_init_pos[0] = ball.pos[0]

            for tile in self.tiles:
                if tile.c != 0:

                    hit, dx, dy, new_ball = tile.get_normal_point(*ball.pos)
                    mag = np.hypot(dx, dy)
                    if hit:
                        ball.reflect(np.array([dx, dy]) / mag)
                        self.hits += 1 + tile.c
                    if new_ball:
                        self.balls.append(Ball(self.ball_init_pos.copy(), dead=True))
                        self.ball_reward += 1
            
            dx, dy = self.get_normal_point(*ball.pos)
            mag = np.hypot(dx, dy)
            if dx != 0 or dy != 0:
                ball.reflect(np.array([dx, dy]) / mag)
    
    def display(self, counter, ang=np.pi/2):
        frame = np.ones((H, W, 3), dtype=np.uint8) * BG
        for tile in self.tiles:
            tile.draw(frame, counter)
        for ball in self.balls:
            ball.draw(frame)
        if self.waiting_for_shot:
            for i in range(2, 10):
                pt2 = self.ball_init_pos + RADIUS * np.array([np.cos(ang), -abs(np.sin(ang))]) * i * 3
                cv2.circle(frame, pt2.astype(int), RADIUS // 2, (230, 230, 230), -1, cv2.LINE_AA)
        return frame
    
    def anim(self, frame):
        self.video.write(frame)

    def close_video(self):
        if self.video is not None:
            self.video.release()
    
    @property
    def reward(self):
        tile_sum = sum([t.c * t.pos[1] for t in self.tiles]) / (6 * len(self.balls))
        hit_sum = self.hits / len(self.balls)
        reward = self.ball_reward * 5 - tile_sum + hit_sum
        return np.clip(reward, -10, 100)
    
    @property
    def state(self):
        arr = np.empty((4, HEIGHT, WIDTH))
        arr[0] = np.array([t.c for t in self.tiles]).reshape(HEIGHT, WIDTH)
        arr[1] = len(self.balls)
        arr[2] = self.ball_init_pos[0] / WIDTH - 0.5
        arr[3] = self.i
        return arr
    
    def step(self, index):
        ang = -(index + 1) * np.pi / (SPLITS + 2)
        self.set_dir(ang)
        key = 0

        frame_counter = 0
        ball_counter = 0
        while not self.all_balls_dead and key != ord("q"):
            if frame_counter % (6 * SKIP) == 0 and ball_counter < len(self.balls):
                self.balls[ball_counter].vel = VEL
                ball_counter += 1
            self.update()
            if frame_counter % SKIP == 0 and self.show_this:
                frame = self.display(frame_counter)
                self.video_frames.append(frame)
                # cv2.imshow("preview", frame)
                # self.anim(frame)
                # key = cv2.waitKey(1)
            frame_counter += 1
        
        reward = self.reward
        done = self.next()
        if done:
            reward = -10

        return self.state, reward, done, key
    
    def display_nick(self):
        frame = self.display(0)
        frame //= 2
        cv2.putText(
            frame,
            str(self.resets),
            (K_TILE, H//2),
            cv2.FONT_HERSHEY_DUPLEX, 
            S_TEXT * 4, 
            (230, 230, 230), 
            T_TEXT * 2,
            cv2.LINE_AA,
        )
        cv2.imshow("preview", frame)
        cv2.waitKey(1)
    
    def reset(self):
        self.close_video()
        for tile in self.tiles:
            tile.c = 0
        self.ball_init_pos = np.array([K_TILE * 7 // 2, K_TILE * 9 - RADIUS], dtype=float)
        self.balls = [Ball(self.ball_init_pos.copy())]
        self.i = 0
        self.hits = 0
        self.ball_reward = 0

        self.next()
        self.resets += 1
        self.show_this = (self.resets % self.show_every) == 1

        if self.show_this:
            self.video_frames = []

        return self.state
    
    def close(self):
        pass

        
if __name__ == "__main__":
    frame_counter = 0
    board = Board()
    board.next()
    cv2.imshow("preview", board.display(frame_counter))
    key = 0

    while key != ord("q"):
        ang = 0
        while key not in [32, ord("q")]:
            if key == 3: # LEFT
                ang += ANG_STEP
                ang = min(-ANG_STEP, ang)
            elif key == 2: # RIGHT
                ang -= ANG_STEP
                ang = max(-np.pi + ANG_STEP, ang)
            cv2.imshow("preview", board.display(frame_counter, ang=ang))
            if frame_counter % SKIP == 0:
                key = cv2.waitKey(1)
            frame_counter += 1

        board.set_dir(ang)

        frame_counter = 0
        ball_counter = 0
        while not board.all_balls_dead and key != ord("q"):
            if frame_counter % (6 * SKIP) == 0 and ball_counter < len(board.balls):
                board.balls[ball_counter].vel = VEL
                ball_counter += 1
            board.update()
            if frame_counter % SKIP == 0:
                cv2.imshow("preview", board.display(frame_counter))
                key = cv2.waitKey(1)
            frame_counter += 1
        
        board.next()
        cv2.imshow("preview", board.display(frame_counter))

    cv2.destroyAllWindows()