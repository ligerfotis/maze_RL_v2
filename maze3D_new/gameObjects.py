from numpy.linalg import norm

from maze3D_new.config import *
from maze3D_new.assets import *
import math
import numpy as np
from scipy.spatial import distance
import time

ball_diameter = 32
damping_factor = 0.4

class GameBoard:
    def __init__(self, layout, discrete=False, rl=False):
        self.velocity = [0, 0]
        self.walls = []
        self.layout = layout
        self.discrete = discrete
        self.rl = rl
        self.scaling_x = 0.05 if self.discrete else 0.01
        self.scaling_y = 0.01 if self.rl else self.scaling_x

        for row in range(len(layout)):
            self.walls.append([])
            for col in range(len(layout[0])):
                self.walls[row].append(None)
                if layout[row][col] != 0:
                    if layout[row][col] == 2:
                        self.hole = Hole(32 * col - 240, 32 * row - 240, self)
                    elif layout[row][col] == 3:
                        self.ball = Ball(32 * col - 240, 32 * row - 240, self)
                    else:
                        self.walls[row][col] = Wall(32 * col - 240, 32 * row - 240, layout[row][col], self)

        self.rot_x = 0
        self.rot_y = 0
        self.max_x_rotation = 0.5
        self.max_y_rotation = 0.5
        self.count_slide = 0
        self.slide = False
        self.slide_velx, self.slide_vely = 0, 0

        self.keyMap = {1: (1, 0),
                       2: (-1, 0),
                       4: (0, 1), 5: (1, 1), 6: (-1, 1), 7: (0, 1),
                       8: (0, -1), 9: (1, -1), 10: (-1, -1), 11: (0, -1), 13: (1, 0), 14: (-1, 0)}

    def getBallCoords(self):
        return (self.ball.x, self.ball.y)

    def collideSquare(self, x, y):
        # if the ball hits a square obstacle, it will return True
        # and the collideTriangle will not be called

        xGrid = math.floor((x + 240) / 32)
        yGrid = math.floor((y + 240) / 32)

        biggest = max(xGrid, yGrid)
        smallest = min(xGrid, yGrid)
        # check the perimeter walls of the tray
        if biggest > 13 or smallest < 1:
            return True, None
        # checks collisons with corner blocks
        if self.walls[yGrid][xGrid] is not None and self.layout[yGrid][xGrid] == 1:
            return True, self.layout[yGrid][xGrid]
        return False, None


    def update(self):
        # compute rotation matrix
        rot_x_m = pyrr.Matrix44.from_x_rotation(self.rot_x)
        rot_y_m = pyrr.Matrix44.from_y_rotation(self.rot_y)
        self.rotationMatrix = pyrr.matrix44.multiply(rot_x_m, rot_y_m)

        self.ball.update()
        self.hole.update()

        for row in self.walls:
            for wall in row:
                if wall != None:
                    wall.update()

    def handleKeys(self, angleIncrement):
        if angleIncrement[0] == 2:
            angleIncrement[0] = -1
        elif angleIncrement[0] == 1:
            angleIncrement[0] = 1

        if angleIncrement[1] == 2:
            angleIncrement[1] = -1
        elif angleIncrement[1] == 1:
            angleIncrement[1] = 1

        self.velocity[0] = self.scaling_y * angleIncrement[0]
        self.rot_x += self.velocity[0]
        if self.rot_x >= self.max_x_rotation:
            self.rot_x = self.max_x_rotation
            self.velocity[0] = 0
        elif self.rot_x <= -self.max_x_rotation:
            self.rot_x = -self.max_x_rotation
            self.velocity[0] = 0

        self.velocity[1] = self.scaling_x * angleIncrement[1]
        self.rot_y += self.velocity[1]
        if self.rot_y >= self.max_y_rotation:
            self.rot_y = self.max_y_rotation
            self.velocity[1] = 0
        elif self.rot_y <= -self.max_y_rotation:
            self.rot_y = -self.max_y_rotation
            self.velocity[1] = 0

    def draw(self, mode=0, idx=0):
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-80, -80, 0]))
        self.model = pyrr.matrix44.multiply(translation, self.rotationMatrix)
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(BOARD_MODEL.getVAO())
        glBindTexture(GL_TEXTURE_2D, BOARD.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, BOARD_MODEL.getVertexCount())

        self.ball.draw()
        self.hole.draw()

        for row in self.walls:
            for wall in row:
                if wall != None:
                    wall.draw()
        # Used for resetting the game. Logs above the board "Game starts in ..."
        if mode == 1:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE,
                               pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            glBindVertexArray(TEXT_MODEL.getVAO())
            glBindTexture(GL_TEXTURE_2D, TEXT[idx].getTexture())
            glDrawArrays(GL_TRIANGLES, 0, TEXT_MODEL.getVertexCount())
        # Used when goal has been reached. Logs above the board "Goal reached"
        elif mode == 2:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE,
                               pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            glBindVertexArray(TEXT_MODEL.getVAO())
            glBindTexture(GL_TEXTURE_2D, TEXT[-2].getTexture())
            glDrawArrays(GL_TRIANGLES, 0, TEXT_MODEL.getVertexCount())
        # Used for resetting the game. Logs above the board "Timeout"
        elif mode == 3:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE,
                               pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            glBindVertexArray(TEXT_MODEL.getVAO())
            glBindTexture(GL_TEXTURE_2D, TEXT[-1].getTexture())
            glDrawArrays(GL_TRIANGLES, 0, TEXT_MODEL.getVertexCount())


class Wall:
    def __init__(self, x, y, type, parent):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        if type in [6, 7]:
            type = 1
        self.type = type - 1

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(WALL_MODELS[self.type].getVAO())
        glBindTexture(GL_TEXTURE_2D, WALL.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, WALL_MODELS[self.type].getVertexCount())


def compute_angle(nextX, nextY):
    if nextX >= 0:
        return np.arctan(nextY / nextX) * 180 / np.pi
    else:
        return 180 + np.arctan(nextY / nextX) * 180 / np.pi


def distance_from_line(p2, p1, p0):
    return  norm(np.cross(p2 - p1, p1 - p0)) / norm(p2 - p1)


class Ball:
    def __init__(self, x, y, parent):
        self.exception = True
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        self.velocity = [0, 0]

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

        # print([self.x, self.y])
        acceleration = [-0.1 * self.parent.rot_y, 0.1 * self.parent.rot_x]
        self.velocity[0] += 1.2 * acceleration[0]
        self.velocity[1] += 1.2 * acceleration[1]

        nextX = self.x + self.velocity[0]
        nextY = self.y + self.velocity[1]

        test_nextX = nextX + ball_diameter / 2 * np.sign(self.velocity[0])
        test_nextY = nextY + ball_diameter / 2 * np.sign(self.velocity[1])

        # check x direction
        checkXCol, gridX = self.parent.collideSquare(test_nextX, self.y)
        checkYCol, gridY = self.parent.collideSquare(self.x, test_nextY)

        if checkXCol:
            self.velocity[0] *= -0.25

        # check y direction
        elif checkYCol:
            self.velocity[1] *= -0.25

        else:
            angle_from_center = compute_angle(nextX, nextY)

            # check if in the upper diagonal barrier
            if -45 <= angle_from_center <= 135:
                # if ball is in the upper triangle of the tray
                self.slide_on_upper_triangle(nextX, nextY, test_nextX, test_nextY)
            elif 135 < angle_from_center <= 180 or angle_from_center <= -45:
                self.slide_on_lower_triangle(nextX, nextY, test_nextX, test_nextY)

        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def draw(self):
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(BALL_MODEL.getVAO())
        glBindTexture(GL_TEXTURE_2D, BALL.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, BALL_MODEL.getVertexCount())

    def slide_on_upper_triangle(self, nextX, nextY, test_nextX, test_nextY):
        # distance of a point (ball's edge towards the move direction) from a line
        p1 = np.asarray([0, 32])
        p2 = np.asarray([32, 0])
        d = norm(np.cross(p2 - p1, p1 - [nextX, nextY])) / norm(p2 - p1)
        if d <= ball_diameter / 2:
            # check if there is an opening
            # print([test_nextX, test_nextY])
            if -16 <= test_nextX <= 48 and -16 <= test_nextY <= 48:
                pass
            elif 16 <= test_nextX <= 48 and test_nextY <= -16:
                if self.velocity[1] <= 0:
                    # bounce on the x axis
                    self.velocity[0] *= damping_factor
                    # keep going on the y axis
                    self.velocity[1] *= -1 * damping_factor
            elif test_nextX <= -16 and 16 <= test_nextY <= 48:
                if self.velocity[0] <= 0:
                    # bounce on the x axis
                    self.velocity[0] *= -1 * damping_factor
                    # keep going on the y axis
                    self.velocity[1] *= damping_factor
            else:
                if self.velocity[0] > 0 and self.velocity[1] < 0:
                    # bounce on the x axis
                    self.velocity[0] = self.velocity[0] + damping_factor * abs(self.velocity[1]) * np.cos(np.pi / 4)
                    # keep going on the y axis
                    self.velocity[1] *= np.sin(np.pi / 4) ** 2
                # go to down right
                elif self.velocity[0] <= 0 and self.velocity[1] <= 0:
                    # keep going on the x axis
                    self.velocity[0] *= -damping_factor
                    # bounce on the y axis
                    self.velocity[1] *= -damping_factor
                # go up
                elif self.velocity[0] <= 0 and self.velocity[1] >= 0:
                    # keep going on the x axis
                    self.velocity[0] *= np.sin(np.pi / 4) ** 2
                    # bounce on the y axis
                    self.velocity[1] = self.velocity[1] + damping_factor * abs(self.velocity[0]) * np.cos(np.pi / 4)

    def slide_on_lower_triangle(self, nextX, nextY, test_nextX, test_nextY):
        # distance of a point (ball's edge towards the move direction) from a line
        p1 = np.asarray([0, -32])
        p2 = np.asarray([-32, 0])
        d = distance_from_line(p2, p1, [nextX, nextY])
        if d <= ball_diameter / 2:
            # check if there is an opening
            # print([test_nextX, test_nextY])
            if -48 <= test_nextX <= 16 and -48 <= test_nextY <= 16:
                pass
            elif 16 <= test_nextX and -48 <= test_nextY <= 16:
                if self.velocity[0] >= 0:
                    # bounce on the x axis
                    self.velocity[0] *= -1 * damping_factor
                    # keep going on the y axis
                    self.velocity[1] *= damping_factor
            elif -48 <= test_nextX <= -16 and 16 <= test_nextY:
                if self.velocity[1] >= 0:
                    # bounce on the x axis
                    self.velocity[0] *= damping_factor
                    # keep going on the y axis
                    self.velocity[1] *= -damping_factor
            else:
                if self.velocity[0] < 0 and self.velocity[1] > 0:
                    # bounce on the x axis
                    self.velocity[0] = self.velocity[0] + damping_factor * abs(self.velocity[1]) * np.cos(np.pi / 4)
                    # keep going on the y axis
                    self.velocity[1] *= np.sin(np.pi / 4) ** 2
                # go to down right
                elif self.velocity[0] >= 0 and self.velocity[1] >= 0:
                    # if in the open space of the frontier
                        # keep going on the x axis
                        self.velocity[0] *= -damping_factor
                        # bounce on the y axis
                        self.velocity[1] *= -damping_factor
                # go up
                elif self.velocity[0] >= 0 and self.velocity[1] <= 0:
                    # keep going on the x axis
                    self.velocity[0] *= np.sin(np.pi / 4) ** 2
                    # bounce on the y axis
                    self.velocity[1] = self.velocity[1] + damping_factor * abs(self.velocity[0]) * np.cos(np.pi / 4)


class Hole:
    def __init__(self, x, y, parent):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(HOLE_MODEL.getVAO())
        glBindTexture(GL_TEXTURE_2D, HOLE.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, HOLE_MODEL.getVertexCount())
