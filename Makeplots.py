from os import F_OK
from PolyTes import PolynomialTessellation
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D

def mk1dsolution():
    mu = np.array((
        [-1,0],
        [1,0]
    ))

    h = 0.001
    x = np.arange(-1.25, 1.25, h)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_xlim([min(x), max(x)])    
    ax2.set_xlim([min(x), max(x)])
    ax3.set_xlim([min(x), max(x)])
    ax1.set_ylim([-0.25, 1.25])
    ax3.set_ylim([-6.5,2.5])

    colors = ['black', 'blue', 'green']
    labels = ['Original', r'$\frac{d^2 g}{d x^2} \geq -6$', r'$\frac{d^2 g}{d x^2} \geq -2$']

    ax1.plot([min(x), max(x)], [0,0], color = 'gray')
    ax2.plot([min(x), max(x)], [0,0], color = 'gray')
    ax3.plot([min(x), max(x)], [0,0], color = 'gray')
    for r,c,l in zip([1, 0.75, 0.5], colors, labels):
        tes = PolynomialTessellation(mu, r)
        f = []
        gds = []
        for _x in x:

            val, gd = tes.eval_fast(np.array((_x, 0)))
            f.append(-val)
            gds.append(-gd)
        f = np.array(f)
        gds = np.array(gds)

        ax1.plot(x, f, color = c, label = l)
        ax2.plot(x, gds[:,0], color = c)
        ax3.plot(x[1:-1], (gds[2:,0] - gds[:-2,0]) / (2*h), color = c)


    ax1.get_yaxis().set_ticks([0])
    ax2.get_yaxis().set_ticks([0])
    ax3.get_yaxis().set_ticks([0])
    ax1.get_xaxis().set_ticks([-1,0,1])
    ax2.get_xaxis().set_ticks([-1,0,1])
    ax3.get_xaxis().set_ticks([-1,0,1])

    ax1.set_title('Artificial Potential (g)')
    ax2.set_title(r'$\frac{dg}{dx}$')
    ax3.set_title(r'$\frac{d^2g}{dx^2}$')

    ax1.legend(loc = 2)


    plt.tight_layout()
    plt.show()


def voronoi_extension():
    mu = np.array((
        [0,1],
        [0.5,0.3],
        [1,1]
    ))

    tess = PolynomialTessellation(mu)


    #voronoi_plot_2d(tess.Vor)

    for verts in tess.Ridge_to_vertex:
        verts = list(verts)
        if len(verts) == 2:
            points = list(set.intersection(*(tess.Vertex_to_point[v] for v in verts)))
            for p in points:
                xs = tess.Vor.points[p] + tess.Ratio * (tess.Vor.vertices[verts] - tess.Vor.points[p])
                plt.plot(*xs.T, color = 'green')
                plt.scatter(*xs.T, color = 'green')
            for v in verts:
                xs = tess.Vor.points[points] + tess.Ratio * (tess.Vor.vertices[v] - tess.Vor.points[points])
                for x in xs:
                    plt.quiver(
                        *tess.Vor.vertices[v], *(x - tess.Vor.vertices[v]), 
                        angles='xy', scale_units='xy', scale=1, facecolor='green', alpha = 0.25)

                for i in range(len(points)):
                    ps = [points[i], points[(i+1) % len(points)]]
                    xs = tess.Vor.points[ps] + tess.Ratio * (tess.Vor.vertices[v] - tess.Vor.points[ps])
                    plt.plot(*xs.T, color = 'green')
            verts = tess.Vor.vertices[verts]
            plt.plot(*verts.T, color = 'blue', zorder = 3)
            plt.scatter(*verts.T, color = 'blue', zorder = 3)
            
            

    plt.scatter(*mu.T, color = 'black')
    plt.xlim([-0.1,1.1])
    plt.ylim([0.2,1.1])
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().set_aspect(1)
    plt.show()


def increased_workspace():
    mu = np.array((
        [0,1],
        [0.5,0.3],
        [1,1]
    ))

    tess = PolynomialTessellation(mu)

    h = 0.01
    s = 0.25
    x = np.arange(0, 1, h)
    y = np.arange(0, 1, h)

    X, Y = np.meshgrid(x,y)
    XY = np.column_stack((X.flatten(), Y.flatten()))
    fs = []
    gds = []
    hs = []
    scale = []
    for xy in XY:
        f, gd, H = tess.eval_fast(xy)
        f = f/2
        H = H/2
        gd = gd/2
        fs.append(np.exp(f / s**2))
        gds.append(gd)
        hs.append(H)

        hess = (np.outer(gd, gd) / s**2 + H) * np.exp(f / s**2)

        scale.append(np.min(np.linalg.eigvals(hess)))

    fs = np.array(fs).reshape(X.shape)
    scale = np.array(scale).reshape(X.shape)
    gds = np.array(gds).reshape((*X.shape, 2))
    hs = np.array(hs).reshape((*X.shape, 2, 2))

    scale = np.clip(scale, -1e5, 1e5)
    #plt.imshow(scale, origin = 1, extent = [0,1,0,1])



    plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot_wireframe(X, Y, fs)
    #ax.plot_wireframe(X, Y, gds[:,:,1])
    plt.show()

def workspace1d():
    mu = np.array((
        [0,0],
        [0.1,0],
        [0.2,0],
        [0.3,0],
        [1,0]
    ))
    #mx = np.arange(0, 1, 0.01)
    #mu = np.column_stack((mx, np.zeros_like(mx)))

    tess = PolynomialTessellation(mu)

    h = 0.0005
    s = 0.1
    x = np.arange(0, 1 + 1e-10, h)
    XY = np.column_stack((x, np.zeros_like(x)))
    fs = []
    gds = []
    for xy in XY:
        f, gd, _ = tess.eval_fast(xy)
        f /= 2
        gd /= 2
        f = np.exp(f / s**2)
        gd *= f
        fs.append(f)
        gds.append(gd[0])
    gds = np.array(gds)
    #plt.plot(x, x + gds)
    #plt.plot(x, x)
    plt.plot(x[1:-1], (gds[2:] - gds[:-2]) / (2 * h))
    plt.show()


def density():
    np.random.seed(1)
    mu = np.array((
        [0.2,0.8],
        [0.5,.2],
        [0.8,0.8]
    ))
    s = 0.5

    tess = PolynomialTessellation(mu)
    XY_p = np.random.random((100000, 2)) * 2 - 0.5

    xs = []
    for xy in XY_p:
        f, gd, H = tess.eval_fast(xy)
        f /= 2
        gd /= 2
        xs.append(xy + np.exp(f/s**2) * gd)
    xs = np.array(xs)

    h = 0.01
    x = np.arange(0, 1, h)
    y = np.arange(0, 1, h)
    X, Y = np.meshgrid(x,y)
    XY = np.column_stack((X.flatten(), Y.flatten()))
    f = []
    sig = 0.05

    #xs = xs.reshape((*xs.shape, 1))
    #XY = XY.T.reshape((1, *XY.T.shape))
    #f = np.sum(np.exp(-np.linalg.norm(xs - XY, axis = 1)**2 / sig**2), axis = 0)
    #f = f.reshape(X.shape)
    plt.scatter(*xs.T, s = 15, alpha = 0.25, edgecolor = 'none')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.figure()
    plt.scatter(*XY_p.T, s = 15, alpha = 0.25, edgecolor = 'none')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.show()


def lines():
    np.random.seed(1)
    mu = np.array((
        [0.2,0.8],
        [0.5,.2],
        [0.8,0.8]
    ))
    s = 0.5

    tess = PolynomialTessellation(mu, 0.8)

    s = 0.2
    h = 0.005
    x = np.arange(-0.5,1.5+1e-10,h)
    y = np.arange(-0.5,1.5+1e-10,h)
    X, Y = np.meshgrid(x, y)

    _XY = np.stack((X, Y), axis = 2)
    
    n = 5
    for i in range(0, len(_XY), n):
        line = []
        for xy in _XY[i,:,:]:
            f, gd, _ = tess.eval_fast(xy)
            f /= 2
            gd /= 2

            line.append(xy + gd/2)# * np.exp(f / s**2))
        line = np.array(line)
        plt.plot(*line.T, color = 'blue')

    for i in range(0, len(_XY), n):
        line = []
        for xy in _XY[:,i,:]:
            f, gd, _ = tess.eval_fast(xy)
            f /= 2
            gd /= 2

            line.append(xy + gd/2)# * np.exp(f / s**2))
        line = np.array(line)

        plt.plot(*line.T, color = 'blue')
    #for i in range(0, len(_XY), n):
    #    plt.plot(_XY[i,:,0], _XY[i,:,1], color = 'blue', linestyle = '--', alpha = 0.3)
    #for i in range(0, len(_XY), n):
    #    plt.plot(_XY[:,i,0], _XY[:,i,1], color = 'blue', linestyle = '--', alpha = 0.3)
    
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.scatter(*mu.T, color = 'red')
    plt.show()

if __name__ == '__main__':
    #mk1dsolution()
    #voronoi_extension()
    #increased_workspace()
    workspace1d()
    #density()
    #lines()


    # Here is the deadline that we are hitting 
    # Here is the plan
    # Here is the demo - why is it impressive
    # Here is the evaluation - what do we care about