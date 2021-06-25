from numpy.linalg import norm

from maze3D_new.config import *
from maze3D_new.assets import *
import math
import numpy as np
from scipy.spatial import distance
import time

ball_diameter = 43.615993
damping_factor = 0.3
discrete_steps_from_center = 5

class GameBoard:
    def __init__(self, layout, discrete=False, rl=False):
        self.box_size = 43.615993
        self.velocity = [0, 0]
        self.walls = []
        self.layout = layout
        self.discrete = discrete
        self.rl = rl
        self.max_x_rotation = 0.5
        self.max_y_rotation = 0.5
        self.scaling_x = self.max_x_rotation / discrete_steps_from_center if self.discrete else 0.03
        self.scaling_y = 0.01 if self.rl else self.scaling_x

        self.num_of_boxes_x = len(layout)
        self.num_of_boxes_y = len(layout[0])

        for row in range(self.num_of_boxes_x):
            self.walls.append([])
            for col in range(self.num_of_boxes_y):
                self.walls[row].append(None)
                if layout[row][col] != 0:
                    if layout[row][col] == 2:
                        self.hole = Hole(self.box_size * col - self.num_of_boxes_y * self.box_size / 2, self.box_size * row - self.num_of_boxes_x * self.box_size / 2, self)
                    elif layout[row][col] == 3:
                        self.ball = Ball(self.box_size * col - self.num_of_boxes_y * self.box_size / 2, self.box_size * row - self.num_of_boxes_x * self.box_size / 2, self)
                    else:
                        self.walls[row][col] = Wall(self.box_size * col - self.num_of_boxes_y * self.box_size / 2, self.box_size * row - self.num_of_boxes_x * self.box_size / 2, layout[row][col], self)

        self.rot_x = 0
        self.rot_y = 0
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

        xGrid = math.floor((x + self.num_of_boxes_x * self.box_size / 2) / self.box_size)
        yGrid = math.floor((y + self.num_of_boxes_y * self.box_size / 2) / self.box_size)

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
        self.box_size = 43.615993

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

        # print([self.x, self.y])
        acceleration = [-0.1 * self.parent.rot_y, 0.1 * self.parent.rot_x]
        self.velocity[0] += 1.5 * acceleration[0]
        self.velocity[1] += 1.5 * acceleration[1]

        nextX = self.x + self.velocity[0]
        nextY = self.y + self.velocity[1]

        test_nextX = nextX + ball_diameter / 2 * np.sign(self.velocity[0])
        test_nextY = nextY + ball_diameter / 2 * np.sign(self.velocity[1])

        # check x direction
        checkXCol, gridX = self.parent.collideSquare(test_nextX, self.y)
        checkYCol, gridY = self.parent.collideSquare(self.x, test_nextY)

        if checkXCol:
            if abs(self.velocity[0]) < 0.1:
                self.velocity[0] = 0
            else:
                self.velocity[0] *= -damping_factor

        # check y direction
        if checkYCol:
            if abs(self.velocity[1]) < 0.1:
                self.velocity[1] = 0
            else:
                self.velocity[1] *= -damping_factor

        angle_from_center = compute_angle(nextX, nextY)

        # check if in the upper diagonal barrier
        if -45 <= angle_from_center <= 135:
            # if ball is in the upper triangle of the tray
            self.slide_on_upper_triangle(nextX, nextY, angle_from_center)
        elif 135 < angle_from_center <= 180 or angle_from_center <= -45:
            self.slide_on_lower_triangle(nextX, nextY, angle_from_center)

        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def draw(self):
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(BALL_MODEL.getVAO())
        glBindTexture(GL_TEXTURE_2D, BALL.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, BALL_MODEL.getVertexCount())

    def slide_on_upper_triangle(self, nextX, nextY, theta):
        # distance of a point (ball's edge towards the move direction) from a line
        p1 = np.asarray([0, self.box_size])
        p2 = np.asarray([self.box_size, 0])
        d = norm(np.cross(p2 - p1, p1 - [nextX, nextY])) / norm(p2 - p1)

        if d <= (ball_diameter / 2):
            # check if there is an opening
            # print([test_nextX, test_nextY])
            if self.velocity[0] > 0 and self.velocity[1] < 0 and theta >= 90:
                if (nextX - ball_diameter/3) > -self.box_size/2:
                    self.velocity *= 1
                    return
                elif (nextX - ball_diameter/2) < -self.box_size/2 and nextY - ball_diameter/2 <= self.box_size*1.5:
                    self.velocity *= 1                                                   
                    return
            if -self.box_size/2 <= nextX - ball_diameter/2 and -self.box_size/2 <= nextY - ball_diameter/2:
                pass
            # block 2
            elif nextX - ball_diameter/2 <= self.box_size*1.5 and (nextY - ball_diameter/2) < -self.box_size/2:
                if self.velocity[1] < 0:
                    # keep going on the y axis
                    self.velocity[1] *= - damping_factor
                    # bounce on the x axis
                    self.velocity[0] = self.velocity[0] + self.velocity[1] * np.sin(theta * np.pi / 180)
            # block 1
            elif (nextX - ball_diameter/2) < -ball_diameter/2 and nextY - ball_diameter/2 <= 48:
                if self.velocity[0] < 0 and self.velocity[1] >= 0:
                    # bounce on the x axis
                    self.velocity[0] *= -1 * damping_factor
                    # keep going on the y axis
                    self.velocity[1] = self.velocity[1] + self.velocity[0] * np.sin(theta * np.pi / 180)
                elif self.velocity[0] < 0 and self.velocity[1] < 0:
                    # bounce on the x axis
                    self.velocity[0] *= -1 * damping_factor
                    # keep going on the y axis
                    self.velocity[1] = self.velocity[1] + self.velocity[0] * np.sin(theta * np.pi / 180)
            # elif self.velocity[0] < 0 and self.velocity[1] > 0 and theta >= 90 and (nextX - ball_diameter/2) < -16 and nextY < 48:
            #         self.velocity[0] = 0.4 * self.velocity[0] + self.velocity[1] * np.cos((-theta) * np.pi / 180)
            #         self.velocity[1] *= np.cos(theta * np.pi / 180) * np.sin((90 - theta) * np.pi / 180)
            else:
                if self.velocity[0] > 0 and self.velocity[1] < 0:
                    # bounce on the x axis
                    if theta > 90:
                        self.velocity[0] = 0.4 * self.velocity[0] + self.velocity[1] * np.cos((-theta) * np.pi / 180)
                        self.velocity[1] *= np.cos(theta * np.pi / 180) * np.sin((90-theta) * np.pi / 180)

                    else:
                        self.velocity[1] *= np.cos(-theta * np.pi / 180) * np.sin((-theta) * np.pi / 180)
                        self.velocity[0] = self.velocity[0] + self.velocity[1] * np.sin((-90+theta) * np.pi / 180)
                        # keep going on the y axis
                # go to down right
                elif self.velocity[0] <= 0 and self.velocity[1] <= 0:
                    if theta > 90 and norm(self.velocity) <= 1.5:
                        self.velocity[0] = self.velocity[1] * np.cos((theta) * np.pi / 180)
                        self.velocity[1] * np.sin(theta * np.pi / 180)
                    elif theta < 0 and norm(self.velocity) <= 1.5:
                        self.velocity[0] = self.velocity[1] * np.cos((180-theta) * np.pi / 180)
                        self.velocity[1] *= np.sin((-theta) * np.pi / 180)
                    else:
                        # keep going on the x axis
                        self.velocity[0] *= -damping_factor
                        # bounce on the y axis
                        self.velocity[1] *= -damping_factor
                # go up
                elif self.velocity[0] <= 0 and self.velocity[1] >= 0:
                    if theta >= 0:
                        # keep going on the x axis
                        self.velocity[0] *= np.sin((90-theta) * np.pi / 180) * np.cos((theta) * np.pi / 180)
                        # bounce on the y axis
                        self.velocity[1] = self.velocity[1] + self.velocity[0] * np.cos(theta * np.pi / 180)
                    else:
                        # keep going on the x axis
                        self.velocity[0] *= np.sin((theta) * np.pi / 180) * np.cos((90-theta) * np.pi / 180)
                        # bounce on the y axis
                        self.velocity[1] = self.velocity[1] + self.velocity[0] * np.cos(
                            90-theta * np.pi / 180)

    def slide_on_lower_triangle(self, nextX, nextY, theta):
        # define the line that the ball must not pass to insert in the frontier
        p1, p2 = np.asarray([0, -self.box_size]), np.asarray([-self.box_size, 0])
        # distance of a point (ball's edge towards the move direction) from a line
        d = distance_from_line(p2, p1, [nextX, nextY])

        # check if the ball's next center position closer than the ball's radius to the frontier line
        if d <= ball_diameter / 2:
            # check if there is an opening
            # print([test_nextX, test_nextY])
            if self.velocity[0] > 0 and self.velocity[1] < 0 and theta >= 180:
                if (nextX + ball_diameter/3) > self.box_size/2:
                    self.velocity *= 1
                    return
                elif (nextX + ball_diameter/2) > self.box_size/2 and nextY - ball_diameter/2 <= 0:
                    self.velocity *= 1
                    return
            if nextX + ball_diameter/2 <= self.box_size/2 and nextY + ball_diameter/2 <= self.box_size/2:
                pass
            # block 2
            elif self.box_size/2 < nextX + ball_diameter/2 and -self.box_size*1.5 <= nextY + ball_diameter/2:
                if self.velocity[0] > 0:
                    # # bounce on the x axis
                    # self.velocity[0] *= -1 * damping_factor
                    # # keep going on the y axis
                    # self.velocity[1] *= damping_factor
                    # bounce on the x axis
                    self.velocity[0] *= -1 * damping_factor
                     # keep going on the y axis
                    self.velocity[1] = self.velocity[1] + self.velocity[0] * np.cos(theta * np.pi / 180)
            # block 1
            elif -self.box_size*1.5 <= nextX + ball_diameter/2 and self.box_size/2 < nextY + ball_diameter/2:
                if self.velocity[1] > 0:
                    # # bounce on the x axis
                    # self.velocity[0] *= damping_factor
                    # # keep going on the y axis
                    # self.velocity[1] *= -damping_factor
                    # bounce on the x axis
                    self.velocity[1] *= -1 * damping_factor
                    self.velocity[0] = self.velocity[0] + self.velocity[1] * np.cos((180-theta) * np.pi / 180)
                    # keep going on the y axis

            else:
                if self.velocity[0] < 0 and self.velocity[1] > 0:
                    if theta < -45:
                        # bounce on the y axis
                        self.velocity[1] *= np.sin(-theta * np.pi / 180) * np.cos((theta) * np.pi / 180)
                        self.velocity[0] = self.velocity[0] + self.velocity[1] * np.sin(-90-theta * np.pi / 180)
                    else:
                        # bounce on the y axis
                        self.velocity[1] *= np.sin(theta * np.pi / 180) * np.cos((180-theta) * np.pi / 180)
                        self.velocity[0] = self.velocity[0] + self.velocity[1] * np.cos(-theta * np.pi / 180)

                # go to down right
                elif self.velocity[0] >= 0 and self.velocity[1] >= 0:
                    # keep going on the x axis
                    # self.velocity[0] *= -damping_factor
                    # # bounce on the y axis
                    # self.velocity[1] *= damping_factor
                    # bounce on the y axis
                    # keep going on the x axis
                    if theta <= -45 and norm(self.velocity) <= 1.5:
                        self.velocity[0] = self.velocity[1] * np.cos((theta) * np.pi / 180)
                        self.velocity[1] *= np.sin((-theta) * np.pi / 180)
                    if theta <= 180 and norm(self.velocity) <= 1.5:
                        self.velocity[1] = self.velocity[0] * np.cos((theta) * np.pi / 180)
                        self.velocity[0] *= np.sin((theta) * np.pi / 180)
                    else:
                        self.velocity[0] *= - damping_factor
                        # bounce on the y axis
                        self.velocity[1] *= damping_factor
                # go up
                elif self.velocity[0] >= 0 and self.velocity[1] <= 0:
                    if theta < -45:
                        self.velocity[0] *= np.sin((90 - theta) * np.pi / 180) * np.cos((theta) * np.pi / 180)
                        self.velocity[1] = self.velocity[1] + self.velocity[0] * np.sin(
                            theta * np.pi / 180)
                    else:
                        # bounce on the y axis
                        self.velocity[1] = self.velocity[0] * np.cos(theta * np.pi / 180)
                        # keep going on the x axis
                        self.velocity[0] *= np.sin(theta * np.pi / 180)


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
