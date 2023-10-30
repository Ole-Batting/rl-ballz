from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    K_TILE, COLORS, WIDTH, HEIGHT, SOBELX, SOBELY, SX, SY, RADIUS, BG, K_TEXT, T_TEXT,
    S_TEXT, B_TILE, SKIP, VEL, H, W
)

pad = RADIUS + VEL
dim = K_TILE + 2 * pad
base_frame = np.zeros((HEIGHT * K_TILE, WIDTH * K_TILE), dtype=np.float32)

normalx = np.zeros((dim, dim), dtype=np.float32)
normalx[pad:-pad, pad:-pad] = 1
normalx = cv2.filter2D(normalx, cv2.CV_32F, SOBELX)
nxn = cv2.dilate(-normalx, SX)
nxp = cv2.dilate(normalx, SX)
normalx = nxp - nxn

normaly = np.zeros((dim, dim), dtype=np.float32)
normaly[pad:-pad, pad:-pad] = 1
normaly = cv2.filter2D(normaly, cv2.CV_32F, SOBELY)
nyn = cv2.dilate(-normaly, SY)
nyp = cv2.dilate(normaly, SY)
normaly = nyp - nyn


def place_hit_box(p1, p2):
    framex = base_frame.copy()
    framey = base_frame.copy()
    pp1 = p1 - pad
    pp2 = p2 + pad
    pp1_ = np.clip(pp1, a_min=0, a_max=None)
    pp2_ = np.clip(pp2, a_min=None, a_max=[W, H])

    y1 = pp1_[1]-pp1[1]
    y2 = pp2_[1]-pp2[1]
    if y2 == 0:
        y2 = framex.shape[0]
    x1 = pp1_[0]-pp1[0]
    x2 = pp2_[0]-pp2[0]
    if x2 == 0:
        x2 = framex.shape[1]
    framex[pp1_[1]:pp2_[1], pp1_[0]:pp2_[0]] = normalx[y1:y2, x1:x2]
    framey[pp1_[1]:pp2_[1], pp1_[0]:pp2_[0]] = normaly[y1:y2, x1:x2]
    return framex, framey


@dataclass
class Tile:
    pos: np.ndarray
    c: int
    normalx: np.ndarray = np.zeros((HEIGHT * K_TILE, WIDTH * K_TILE), dtype=np.uint8)
    normaly: np.ndarray = np.zeros((HEIGHT * K_TILE, WIDTH * K_TILE), dtype=np.uint8)

    @property
    def p1(self):
        return self.pos * K_TILE
    
    @property
    def p2(self):
        return (self.pos + 1) * K_TILE

    @property
    def pt(self):
        return ((self.pos + [0.5, 0.5]) * K_TILE).astype(int)
    
    @property
    def color(self):
        b = np.interp(self.c, COLORS[:, 3], COLORS[:, 0])
        g = np.interp(self.c, COLORS[:, 3], COLORS[:, 1])
        r = np.interp(self.c, COLORS[:, 3], COLORS[:, 2])
        return int(b), int(g), int(r)

    def draw(self, frame, counter):
        if self.c > 0:
            s = 1
            cv2.rectangle(frame, self.p1, self.p2, self.color, -1)
            cv2.rectangle(frame, self.p1, self.p2, (BG, BG, BG), B_TILE)
            size = np.array(cv2.getTextSize(str(self.c), cv2.FONT_HERSHEY_DUPLEX, S_TEXT, T_TEXT)[0])
            pt = (self.pt + np.array([-1, 1]) * s * size // 2).astype(int)
            if size[0] > K_TEXT:
                s =  K_TEXT / size[0]
            cv2.putText(
                frame,
                str(self.c),
                pt,
                cv2.FONT_HERSHEY_DUPLEX, 
                S_TEXT * s, 
                (BG, BG, BG), 
                T_TEXT,
                cv2.LINE_AA,
            )
        elif self.c == -1:
            s = np.cos(counter / (3 * SKIP)) * 0.2 + 2
            cv2.circle(frame, self.pt, int(RADIUS * s), (230, 230, 230), T_TEXT * 2, cv2.LINE_AA)
            cv2.circle(frame, self.pt, RADIUS, (230, 230, 230), -1, cv2.LINE_AA)
    
    def _set_hit_box(self):
        if self.c != 0:
            frame = np.zeros((HEIGHT * K_TILE, WIDTH * K_TILE), dtype=np.float32)
            if self.c == -1:
                cv2.circle(frame, self.pt, RADIUS * 2, (1), -1)
            else:
                cv2.rectangle(frame, self.p1, self.p2, (1), -1)

            normalx = cv2.filter2D(frame, cv2.CV_32F, SOBELX)
            nxn = cv2.dilate(-normalx, SX)
            nxp = cv2.dilate(normalx, SX)
            self.normalx = nxp - nxn

            normaly = cv2.filter2D(frame, cv2.CV_32F, SOBELY)
            nyn = cv2.dilate(-normaly, SY)
            nyp = cv2.dilate(normaly, SY)
            self.normaly = nyp - nyn
    
    def set_hit_box(self):
        if self.c != 0:
            self.normalx, self.normaly = place_hit_box(self.p1, self.p2)
        
    def get_normal_point(self, x, y):
        dx = self.normalx[int(y), int(x)]
        dy = self.normaly[int(y), int(x)]
        hit = dx != 0 or dy != 0
        new_ball = False
        if hit:
            self.c -= 1
            if self.c == 0:
                self.set_hit_box()
            if self.c == -2:
                self.c = 0
                new_ball = True
                hit = False
        return hit, dx, dy, new_ball
    
    def _get_normal_point(self, x, y):
        hit, dx, dy = self.get_normal_point_1(x, y)
        hit, new_ball = self.get_normal_point_2(hit)
        return hit, dx, dy, new_ball
    
    def get_normal_point_1(self, x, y):
        dx = self.normalx[int(y), int(x)]
        dy = self.normaly[int(y), int(x)]
        hit = dx != 0 or dy != 0
        return hit, dx, dy
    
    def get_normal_point_2(self, hit):
        new_ball = False
        if hit:
            self.c -= 1
            if self.c == 0:
                self.set_hit_box()
            if self.c == -2:
                self.c = 0
                new_ball = True
                hit = False
        return hit, new_ball
    
    def roll(self):
        self.pos[1] += 1
        self.set_hit_box()

if __name__ == "__main__":
    frame = np.ones((HEIGHT * K_TILE, WIDTH * K_TILE, 3), dtype=np.uint8) * BG
    tile = Tile(np.array([0, 0]), 20)
    tile.draw(frame)
    print(frame.dtype, frame.shape)
    cv2.imshow("preview", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
