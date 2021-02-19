from __future__ import barry_as_FLUFL
import numpy as np
from numpy.lib.function_base import bartlett
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import itertools
from collections import defaultdict
from scipy.spatial import Voronoi, voronoi_plot_2d
import cdd
import scipy
from queue import Queue

def eval_poly(x, alpha):
    n = 2
    return np.sum(row(x, n) * alpha)
    
def grad_poly(x, alpha):
    n = 2
    return np.sum(rows_dx(x,n) * alpha, axis = 1)


def row(x, n):
    r = []
    for i in range(n+1):
        for xs in itertools.combinations_with_replacement(x, i):
            r.append(np.prod(xs))
    return np.array(r)


def rows_dx(x, n):
    rs = []
    for x_i, v in enumerate(x):
        r = []
        for i in range(n+1):
            for x_is in itertools.combinations_with_replacement(range(len(x)), i):
                if x_i not in x_is:
                    r.append(0)
                else:
                    xs = x[list(x_is)]
                    xs[x_is.index(x_i)] = x_is.count(x_i)
                    r.append(np.prod(xs))
        rs.append(r)
    return np.array(rs)

 
def fit_poly(xs, fs, dfs, n = None):
    A = []
    b = []
    
    if n is None:
        n = required_order(len(fs) + len(xs[0])*len(dfs))
    
    for x, f, df in zip(xs, fs, dfs):
        A.append(row(x, n))
        b.append(f)
        
        A.extend(rows_dx(x, n))
        b.extend(df)
        
    A = np.array(A)
    b = np.array(b)
    a = np.linalg.lstsq(A, b, rcond = None)[0]
    return a, np.max(np.abs(A @ a - b))


class PolynomialTessellation:
    def __init__(self, mu, ratio = 0.5, max_dist = 20):
        self.Ratio = ratio
        max_dist = 20
        self.InitialCount = len(mu)
        boundaries = []
        for combo in itertools.product(*(((-1.,1.),) * len(mu[0]))):
            boundaries.append(np.array(combo) * max_dist)
        
        print('voronoi')
        mu = np.concatenate((mu, boundaries))
        self.Vor = Voronoi(mu)
        print('done voronoi')
        self.Point_to_point = defaultdict(set)
        self.Vertex_to_point = defaultdict(set)
        self.Point_to_vertex = defaultdict(set)
        self.Ridge_to_vertex = []
        self.Point_to_ridge = defaultdict(set)
        self.Point_point_to_ridge = {}
        for r, (rps, rvs) in enumerate(zip(self.Vor.ridge_points, self.Vor.ridge_vertices)):
            self.Ridge_to_vertex.append(set(v for v in rvs if v != -1))
            for i in rps:
                if r not in self.Point_to_ridge[i]:
                    self.Point_to_ridge[i].add(r)
                self.Point_point_to_ridge[frozenset(rps)] = r
                self.Point_to_point[i] |= set(j for j in rps if j != i)
                for j in rvs:
                    if j > -1 and i not in self.Vertex_to_point[j]:
                        self.Vertex_to_point[j].add(i)
                    if j > -1 and j not in self.Point_to_vertex[i]:
                        self.Point_to_vertex[i].add(j)

        self.PointBoundaries = []
        for i, p in enumerate(self.Vor.points):
            A = []
            b = []
            adj = []
            for a in self.Point_to_point[i]:
                n = self.Vor.points[a] - p
                r = np.linalg.norm(n) / 2
                n /= 2*r
                A.append(n)
                b.append(r)
                adj.append(a)
            A = np.array(A)
            b = np.array(b)
            self.PointBoundaries.append((i,A,b,adj))
        print('tessellated')

    def neighbor_regions(self, point_combo, dimension):
        if len(point_combo) > 1:
            for combo in itertools.combinations(point_combo, len(point_combo) - 1):
                yield combo
        if len(point_combo) <= dimension:
            possible_additions = set.intersection(*(self.Point_to_point[p] for p in point_combo))
            for a in possible_additions:
                yield (*point_combo, a)
        return None
    

    def new_direction(self, region, new_p):
        points = np.array([self.Vor.points[p] for p in region]).reshape((len(region), self.Vor.points.shape[1]))
        next = self.Vor.points[new_p]

        next = next - points[0,:]
        points = points - points[0,:]
        directions = []
        for new_d in points[1:,:]:
            new_d /= np.linalg.norm(new_d)
            for d in directions:
                new_d -= new_d @ d * d
                new_d /= np.linalg.norm(new_d)
            directions.append(new_d)
        
        next /= np.linalg.norm(next)
        for d in directions:
            next -= next @ d * d
            next /= np.linalg.norm(next)
        return next

    def should_move(self, region, neighbor, x):
        region_v = set.intersection(*(self.Point_to_vertex[p] for p in region))
        neighbor_v = set.intersection(*(self.Point_to_vertex[p] for p in neighbor))
        region = set(region)
        neighbor = set(neighbor)
        shared_verts = region_v & neighbor_v
        plane_verts = []
        shared_points = region & neighbor
        n = None
        for v in shared_verts:
            for p in shared_points:
                plane_verts.append(self.Vor.points[p] + self.Ratio * (self.Vor.vertices[v] - self.Vor.points[p]))
                if len(plane_verts) >= len(x) + 1:
                    A = np.array(plane_verts[1:])
                    A -= plane_verts[0]#np.mean(A, axis = 0)
                    U, s, V = np.linalg.svd(A)
                    if s[-1] > 1e-12:
                        print('Ahhh')
                    if np.sum(s < 1e-12) == 1:
                        n = V[-1]
                        break
            if n is not None:
                break
        
        if len(plane_verts) == 0:
            return False

        if n is None:
            A = np.array(plane_verts)
            A -= np.mean(A, axis = 0)
            U, s, V = np.linalg.svd(A)
            if s[-1] > 1e-10:
                print('Ahhh')
            n = V[-1]

        if np.sum(s < 1e-10) > 1:
            return False
        
        x0 = plane_verts[-1]
        b = n @ x0

        rb = 0
        for v in region_v:
            for p in region:
                rb = n @ (self.Vor.points[p] + self.Ratio * (self.Vor.vertices[v] - self.Vor.points[p])) - b
                if np.abs(rb) > 1e-8:
                    break
            if np.abs(rb) > 1e-8:
                break
        xb = n @ x - b

        if np.abs(xb) <= 1e-12:
            return False
        if np.sign(xb) == np.sign(rb):
            return False
        return True

    def should_move_fast(self, region, neighbor, x, region_verts):
        region = set(region)
        neighbor = set(neighbor)
        new_point = list(region ^ neighbor)
        if len(new_point) != 1:
            print(new_point)
            print('ahhh')
        n = self.new_direction(region & neighbor, new_point[0])

        vals = region_verts @ n
        xn = n.dot(x)
        return xn < np.min(vals) or xn > np.max(vals)
        

    def quick_contains(self, region, x):
        if len(region) == 1:
            p = region[0]
            _, A, b, _ = self.PointBoundaries[p]
            return np.all(A @ (x - self.Vor.points[p]) <= b*self.Ratio)
        return False

    
    def eval_fast(self, x):
        p_i = np.argmin(np.linalg.norm(self.Vor.points - x, axis = 1))
        best_region = (p_i,)
        while True:
            if self.quick_contains(best_region, x):
                return self.eval_region(best_region, x)
            pvs = itertools.product(best_region, set.intersection(*(self.Point_to_vertex[p] for p in best_region)))
            ps, vs = list(zip(*pvs))
            ps = list(ps)
            vs = list(vs)
            region_verts = self.Vor.points[ps] + self.Ratio * (self.Vor.vertices[vs] - self.Vor.points[ps])


            for neighbor in self.neighbor_regions(best_region, len(x)):
                if self.should_move_fast(best_region, neighbor, x, region_verts):
                    best_region = neighbor
                    break

                #if self.should_move(best_region, neighbor, x):
                #    best_region = neighbor
                #    break
            else:
                return self.eval_region(best_region, x)

    
    def eval_region(self, region, x):
        if len(region) == 1:
            p = region[0]
            point = self.Vor.points[p]
            return -(x - point) @ (x - point), - 2 * (x - point)
        
        vertices = set.intersection(*(self.Point_to_vertex[p] for p in region))
        p = region[0]
        x0 = self.Vor.points[p] + self.Ratio * (self.Vor.vertices[vertices[0]] - self.Vor.points[p])
        f0 = -(x0-p)@(x0-p)
        r = []
        vps = []
        for v_i in vertices:
            for p_i in region:
                p = self.Vor.points[p_i]
                a = p + self.Ratio * (self.Vor.vertices[v_i] - p)
                value = (
                    a,
                    -(a-p)@(a-p),
                    -2*(a - p)
                )
                r.append(value)
                vps.append((v_i, p_i))
        if len(r) == 0:
            return np.inf, None, None, None
        mat = []
        xs = []
        fs = []
        dfs = []
        for a,f,df in r:
            mat.append(np.concatenate(([1], a), axis = 0))
            xs.append(a)
            fs.append(f)
            dfs.append(df)
        
        xs = np.array(xs)
        #coeffs, err = fit_poly(xs[:len(x)*3], fs[:len(x)*3], dfs[:len(x)*3], 2)
        points = np.array([self.Vor.points[p] for p in region])
        A, b, c, p = fast_q_fit(xs, fs, dfs, points, self.Ratio, np.mean(xs, axis = 0), xs[0], fs[0])


        #print('comparison')
        #print(abs(eval_poly(x, coeffs) - ((x - p) @ A @ (x - p) + b @ (x - p) + c)))
        #print(np.max(np.abs(grad_poly(x, coeffs) - (2 * A @ (x - p) + b))))
        #return eval_poly(x, coeffs), grad_poly(x, coeffs)
        return (x - p) @ A @ (x - p) + b @ (x - p) + c, 2 * A @ (x - p) + b

def fast_q_fit(points, ratio, mean_v, rx, rf):
    p = np.mean(points, axis = 0)
    points = points[1:] - points[0]
    D = points.T
    
    orth = scipy.linalg.orth(D, rcond = None)
    null = scipy.linalg.null_space(D.T, rcond = None)

    s = np.diag([ratio / (1-ratio)] * len(orth.T) + [-1] * len(null.T))
    V = np.column_stack((orth, null))
    A = V @ s @ V.T

    b = -2*(mean_v - p) - 2 * A @ (mean_v - p)

    c = rf - ((rx - p) @ A @ (rx - p) + b @ (rx - p))

    return A, b, c, p

def show():
    mu = np.random.random((5, 2))
    mu = np.column_stack((mu, np.zeros(mu.shape[0])))
    tess = PolynomialTessellation(mu, 0.5)


    h = 0.025

    x = np.arange(0,1,h)
    y = np.arange(0,1,h)
    X, Y = np.meshgrid(x,y)
    XY = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))

    f = []
    gds = []
    for xy in XY:
        v, gd = tess.eval_fast(xy)
        f.append(v)
        gds.append(gd/2)
    f = np.array(f).reshape(X.shape)
    gds = np.array(gds)#.reshape(X.shape)


    ax = plt.gca(projection = '3d')
    ax.plot_wireframe(X, Y, f)
    plt.axis('off')

    plt.figure()
    plt.quiver(X.flatten(), Y.flatten(), gds[:,0], gds[:,1], angles='xy', scale_units='xy', scale=1)
    plt.axis('off')

    plt.show()

def perf_test():
    from cProfile import Profile
    from pstats import Stats
    d = 6
    mu = np.random.random((10, d))
    tess = PolynomialTessellation(mu, 0.5)

    p = Profile()
    p.enable()
    for i in range(100):
        tess.eval_fast(np.random.random(d) * 2 - 1)
    p.disable()
    Stats(p).sort_stats('cumtime').print_stats()

def small_test():
    mu = np.array([
        [0,0.5],
        [0.5,0],
        [1,0.5]
    ])
    tess = PolynomialTessellation(mu, 0.75)

    xs = np.array((
        [0.7, 0.4],
        [0.5, 0.5]
    ))

    for x in xs:
        tess.eval_fast(x)

if __name__ == '__main__':

    np.random.seed(1)
    perf_test()
    #show()
    #small_test()


