import cv2
import numpy as np

WIDTH = 7
HEIGHT = 9
K_RES = 10
K_TILE = 8 * K_RES
B_TILE = K_RES * 3 // 4
RADIUS = K_RES
W = WIDTH * K_TILE
H = HEIGHT * K_TILE
K_TEXT = 6 * K_RES
T_TEXT = max(1, K_RES // 10)
S_TEXT = 1.2 * K_RES / 10
X_BALL = W // 2
Y_BALL = H - RADIUS
SKIP = 2
ANG_STEP = 0.04
SPLITS = 24

COLORS = np.array([
    [0, 0, 0, 0],
    [0, 185, 235, 1],
    [0, 235, 150, 16],
    [50, 10, 245, 32],
    [150, 0, 200, 64],
    [245, 50, 10, 128],
    [150, 235, 0, 256],
])
BG =  30

P = 0.61
Q = 0.21
VEL = 6 * K_RES // 10

SOBELX = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
])
SOBELY = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1,-2,-1],
])
SX = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RADIUS + VEL, (RADIUS + VEL) // 2))
SY = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((RADIUS + VEL) // 2, RADIUS + VEL))