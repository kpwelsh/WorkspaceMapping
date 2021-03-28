import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import itertools
from collections import defaultdict
from scipy.spatial import Voronoi, voronoi_plot_2d
import cdd
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
            for x_is in itertools.combinations_with_replacement(range(len(x)),  i):
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


    def get_vertex_combos(self, x):
        relevant_points = set()
        for p_i, A, b, adj in self.PointBoundaries:
            p = self.Vor.points[p_i]
            d = A @ (x - p)
            if np.all(d <= b*self.Ratio):
                return None, relevant_points, -(x-p)@(x-p), -2*(x-p), 1
            # if np.all(d <= b):
            #     if p_i not in relevant_points:
            #         relevant_points.add(p_i)
            #     for i in range(len(b)):
            #         if d[i] >= 0:
            #             if adj[i] not in relevant_points:
            #                 relevant_points.add(adj[i])
        relevant_points = set(range(len(self.Vor.points)))
        
        adjacencies = [(frozenset({p}), self.Point_to_point[p]) for p in relevant_points]
        vertex_combos = set()
        for i in range(2, len(x) + 2):
            next_point_combos = set()
            next_adjacencies = []
            for combo, adj in adjacencies:
                for a in adj:
                    if a in relevant_points:
                        next_combo = frozenset.union(combo,frozenset({a}))
                        if next_combo not in next_point_combos:
                            next_point_combos.add(next_combo)
                            next_adjacencies.append((next_combo, adj & self.Point_to_point[a]))
            adjacencies = next_adjacencies
            for combo in next_point_combos:
                vs = set.intersection(*(self.Point_to_vertex[p_i] for p_i in combo))
                vs = frozenset(vs)
                if len(vs) > 0 and vs not in vertex_combos:
                    vertex_combos.add(vs)
        #print(len(relevant_points), len(vertex_combos))
        return vertex_combos, relevant_points, None

    def get_vertex_combos_fast(self, x):
        p_i = np.argmin(np.linalg.norm(self.Vor.points - x, axis = 1))
        relevant_points = {p_i}
        q = Queue()
        q.put(p_i)
        while not q.empty():
            p = q.get()
            _, A, b, adj = self.PointBoundaries[p]
            violations = A @ (x - self.Vor.points[p]) - b * self.Ratio
            if np.all(violations <= 0):
                return None, set(), -(x-p)@(x-p), -2*(x-p)
            for v, a in zip(violations, adj):
                if v > 0 and a not in relevant_points:
                    relevant_points.add(a)
                    q.put(a)

        adjacencies = [(frozenset({p}), self.Point_to_point[p]) for p in relevant_points]
        vertex_combos = set()
        for i in range(2, len(x) + 2):
            next_point_combos = set()
            next_adjacencies = []
            for combo, adj in adjacencies:
                for a in adj:
                    if a in relevant_points:
                        next_combo = frozenset.union(combo,frozenset({a}))
                        if next_combo not in next_point_combos:
                            next_point_combos.add(next_combo)
                            next_adjacencies.append((next_combo, adj & self.Point_to_point[a]))
            adjacencies = next_adjacencies
            for combo in next_point_combos:
                vs = set.intersection(*(self.Point_to_vertex[p_i] for p_i in combo))
                vs = frozenset(vs)
                if len(vs) > 0 and vs not in vertex_combos:
                    vertex_combos.add(vs)
        return vertex_combos, relevant_points, None

    def neighbor_regions(self, point_combo, dimension):
        if len(point_combo) > 1:
            for combo in itertools.combinations(point_combo, len(point_combo) - 1):
                yield combo
        if len(point_combo) <= dimension:
            possible_additions = set.intersection(*(self.Point_to_point[p] for p in point_combo))
            for a in possible_additions:
                yield (*point_combo, a)
        return None

    def contains(self, points, x):
        if len(points) == 1:
            p = points[0]
            _, A, b, _ = self.PointBoundaries[p]
            d = self.distance(A, b * self.Ratio, (x - self.Vor.points[p]))
            return d, -(x-self.Vor.points[p])@(x-self.Vor.points[p]), -2*(x-self.Vor.points[p])
        
        vertices = set.intersection(*(self.Point_to_vertex[p] for p in points))
        r = []
        for v_i in vertices:
            for p_i in points:
                p = self.Vor.points[p_i]
                a = p + self.Ratio * (self.Vor.vertices[v_i] - p)
                value = (
                    a,
                    -(a-p)@(a-p),
                    -2*(a - p)
                )
                r.append(value)
        if len(r) == 0:
            return np.inf, None
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
        mat = cdd.Matrix(mat, number_type = 'float')
        mat.rep_type = cdd.RepType.GENERATOR
        try:
            poly = cdd.Polyhedron(mat)
        except RuntimeError as e:
            print(e)
            return np.inf, None
        bA = np.array(poly.get_inequalities(), dtype = np.float64)
        b = bA[:,0]
        A = -bA[:,1:]
        scale = np.linalg.norm(A, axis = 1)
        b /= scale
        A = (A.T / scale).T
        d = self.distance(A, b, x)
        if d <= 1e-10:
            coeffs, err = fit_poly(xs, fs, dfs, 2)
            print(err)
            return d, eval_poly(x, coeffs), grad_poly(x, coeffs)
        return d, None

    def eval_fast(self, x):
        p_i = np.argmin(np.linalg.norm(self.Vor.points - x, axis = 1))
        points = (p_i,)
        m, *result = self.contains(points, x)
        if m <= 1e-10:
            return result
        best_m = m
        best_region = points
        while True:
            for neighbor in self.neighbor_regions(best_region, len(x)):
                m, *result = self.contains(neighbor, x)
                if m <= 1e-10:
                    return result
                
                if m < best_m:
                    best_m = m
                    best_region = neighbor
        
    def distance(self, A, b, x):
        res = minimize(lambda a: (a-x) @ (a-x), x.copy(), method = 'SLSQP',
                        constraints = [LinearConstraint(A, -np.inf, b)])
        return np.sqrt(res['fun'])

    def eval(self, x):
        vertex_combos, relevant_points, *other = self.get_vertex_combos(x)
        if vertex_combos is None:
            return other
        anchor_cache = {}
        compute_time = 0
        for ii, combo in enumerate(sorted(vertex_combos, key = len, reverse = True)):
            points = set.intersection(*(self.Vertex_to_point[v] for v in combo))
            if np.all(np.array(list(points)) >= self.InitialCount):
                continue
            r = []
            for v_i in combo:
                for p_i in points:
                    key = (v_i, p_i)
                    if key in anchor_cache:
                        r.append(anchor_cache[key])
                    else:
                        p = self.Vor.points[p_i]
                        a = p + self.Ratio * (self.Vor.vertices[v_i] - p)
                        value = (
                            a,
                            -(a-p)@(a-p),
                            -2*(a - p)
                        )
                        r.append(value)
                        anchor_cache[key] = value
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
            if np.any(np.min(xs, axis = 0) > x):
                continue
            if np.any(np.max(xs, axis = 0) < x):
                continue
            
            mat = cdd.Matrix(mat, number_type = 'float')
            mat.rep_type = cdd.RepType.GENERATOR
            try:
                poly = cdd.Polyhedron(mat)
            except RuntimeError as e:
                print(e)
                continue
            bA = np.array(poly.get_inequalities(), dtype = np.float64)
            b = bA[:,0]
            A = -bA[:,1:]
            if np.all(A @ x <= b):
                coeffs, err = fit_poly(xs[:len(x)*3], fs[:len(x)*3], dfs[:len(x)*3], 2)
                #print(points)
                #print('correct points')
                for p in points:
                    _, A, b, _ = self.PointBoundaries[p]
                    #print(np.sum(A @ (x - self.Vor.points[p]) - b * self.Ratio > 0))
                #print('incorrect points')
                for p in relevant_points - points:
                    _, A, b, _ = self.PointBoundaries[p]
                    #print(np.sum(A @ (x - self.Vor.points[p]) - b * self.Ratio > 0))

                return eval_poly(x, coeffs), grad_poly(x, coeffs), 1
        raise ValueError
        return 1, np.zeros_like(x), 1




def show():
    mu = np.array((
        [0,0.25],
        [0.5,0],
        [1,0.25]
    ))
    tess = PolynomialTessellation(mu, 0.5)


    h = 0.05

    x = np.arange(0,1,h)
    y = np.arange(0,1,h)
    X, Y = np.meshgrid(x,y)
    XY = np.column_stack((X.flatten(), Y.flatten()))

    f = []
    for xy in XY:
        v, gd = tess.eval_fast(xy)
        f.append(v)
    f = np.array(f).reshape(X.shape)


    ax = plt.gca(projection = '3d')
    ax.plot_wireframe(X, Y, f)
    plt.show()




if __name__ == '__main__':

    import time
    from cProfile import Profile
    from pstats import Stats
    np.random.seed(1)
    d = 5
    mu = np.random.random((25, d))
    tess = PolynomialTessellation(mu, 0.5)

    p = Profile()
    p.enable()
    for i in range(100):
        tess.eval(np.random.random(d))
        print(time.time())
    p.disable()
    Stats(p).sort_stats('cumtime').print_stats()