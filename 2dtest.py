from numpy.lib.shape_base import column_stack
import pygame
import numpy as np

from PolyTes import PolynomialTessellation

class QuadraticTesselation:
    def __init__(self, targets):
        smoothness = 0.5 # within (0.0, 1.0)
        self.effect_scale = 0.75
        self.s = 0.1
        self.effect_function = lambda gradient, function_value : gradient * np.exp(function_value/s**2)

        self.tess = PolynomialTessellation(targets, 1 - smoothness)

    def __call__(self, dx, pos, true_x):
        f, gd, _ = self.tess.eval_fast(true_x)
        f /= 2
        gd /= 2
        return true_x + self.effect_function(gd, f) * self.effect_scale

class SemanticPointing:
    def __init__(self, targets):
        self.Targets = targets
        self.Width = 0.05
        self.S = 0
    
    def __call__(self, dx, pos, true_x):
        if np.linalg.norm(dx) < 1e-10:
            return pos
        return pos + self.scale(pos) * dx

    def omega(self, u):
        return np.log(3) / np.cosh(np.log(3) * u)
    
    def scale(self, pos):
        s = 1 - self.omega(np.min(np.linalg.norm(pos - self.Targets, axis = 1)) / self.Width)
        s = max(s, 0.2)
        return s

class SlowNearPoints:
    def __init__(self, targets):
        self.Targets = targets
        self.S = 0.1
    
    def __call__(self, dx, pos, true_x):
        if np.linalg.norm(dx) < 1e-10:
            return pos
        return pos + self.scale(pos) * dx

    
    def scale(self, pos):
        return 1 - 0.7*np.exp(-np.min(np.linalg.norm(pos - self.Targets, axis = 1))**2 / self.S **2)

class Acceleration:
    def __init__(self, *args):
        pass

    def __call__(self, dx, pos, true_x):
        if np.linalg.norm(dx) < 1e-8:
            return pos
        return pos + dx * np.linalg.norm(dx) / 0.010

class Momentum:
    def __init__(self, *args):
        self.V = 0
    
    def __call__(self, dx, pos, true_x):
        self.V *= 0.6
        self.V += dx*0.5
        return pos + self.V

class Nothing:
    def __init__(self, *args):
        pass
    def __call__(self, dx, pos, true_x):
        return pos + dx

class Hysterysis:
    def __init__(self, *args):
        pass
    def __call__(self, dx, pos, true_x):
        return pos + dx

class SnapDragging:
    #heres why the graphics solutions wont work for robots.
    #Heres why the robot solutions might work for graphics
    
    # Sophisticated systems like Gargoyle

    # Is there a 6d non robot application. Some 3d modeling thing?


    pass



if __name__ == '__main__':
    screen_size = np.array([1000,1000])
    smoothness = 0.5 # within (0.0, 1.0)
    effect_scale = 0.75
    s = 0.1
    effect_function = lambda gradient, function_value : gradient * np.exp(function_value/s**2)


    pygame.init()
    screen = pygame.display.set_mode(screen_size)

    np.random.seed(2)
    mu = np.random.random((10, 2))
    #mu = np.array((
    #    [0.25,0.5],
    #    [0.55,0.5],
    #    [0.75,0.5],
    #))
    #mx = np.arange(0, 1, 0.025)
    #mu = np.column_stack((mx,0.5 * np.ones_like(mx)))

    methods = [
        QuadraticTesselation,
        SemanticPointing,
        SlowNearPoints,
        Acceleration,
        Momentum,
        Nothing
    ]


    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    true_x = np.array(pygame.mouse.get_pos(), dtype = np.float) / screen_size
    cursor = np.array(pygame.mouse.get_pos(), dtype = np.float) / screen_size
    method_index = 0
    method = methods[method_index](mu)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    method_index = (method_index + 1) % len(methods)
                    print(methods[method_index])
                    method = methods[method_index](mu)
                    true_x = cursor.copy()
        else:
            screen.fill((255,255,255))

            for m in mu:
                pygame.draw.circle(screen, (0,0,255), np.array(m * screen_size, dtype = np.int), 10)

            dx = np.array(pygame.mouse.get_rel(), dtype = np.float) / screen_size
            true_x += dx
            cursor = method(dx, cursor, true_x)
            cursor = np.clip(cursor, 0, 1)

            pygame.draw.circle(screen, (255,0,0), cursor * screen_size, 3)

            #pygame.draw.circle(screen, (0,0,0), true_x * screen_size, 3)
            #draw_arrow(screen, (0,0,0), true_x * screen_size, cursor * screen_size)

            pygame.display.flip()
            continue
        break
    pygame.quit()

