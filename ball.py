from dataclasses import dataclass

import cv2
import numpy as np

from constants import RADIUS, W, H, VEL, Y_BALL

@dataclass
class Ball:
    pos: np.ndarray
    dir: np.ndarray = np.array([0, 0], dtype=float)
    vel: float = 0
    dead: bool = True

    def update(self):
        if not self.dead:
            self.pos += self.dir * self.vel
            if self.pos[1] > H - RADIUS - VEL and self.dir[1] > 0:
                self.dead = True
                self.pos[1] = Y_BALL
                return True
            self.pos[1] = max(self.pos[1], 0)
            self.pos[0] = max(self.pos[0], 0)
            self.pos[0] = min(self.pos[0], W-1)
        return False
    
    def draw(self, frame):
        if not self.dead:
            cv2.circle(
                frame, self.pos.astype(int), RADIUS, (230, 230, 230), -1, cv2.LINE_AA,
            )
    
    def reflect(self, normal):
        dot = self.dir @ normal
        if dot < 0:
            conormal = dot * normal
            self.dir -= 2 * conormal
            self.dir[1] += 0.0005
            self.dir /= np.hypot(*self.dir)