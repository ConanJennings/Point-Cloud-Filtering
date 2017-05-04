import random
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D

import pickle as cPickle


def load_data(filename):
    res = [[] for _ in range(3)]
    with open(filename, 'r') as infile:
        for line in infile.readlines():
            _, x, y, z = line.split()
            x, y, z = map(float, [x, y, z])
            res[0].append(x)
            res[1].append(y)
            res[2].append(z)
    return map(np.array, res)


def neighbors(x, y, size, x_min, x_max, y_min, y_max):
    res = []
    for i in range(max(x_min, x - size), min(x_max + 1, x + size + 1)):
        for j in range(max(y_min, y - size), min(y_max + 1, y + size + 1)):
            if not (x == i and y == j):
                res.append([i, j])
    return res


def clean_data(lat, lon, alt, grid_threshold=0.8):
    order = np.argsort(lat)
    ordered_lat, ordered_lon, ordered_alt = lat[
        order], lon[order], alt[order]

    min_lat, max_lat = min(lat), max(lat)
    min_lon, max_lon = min(lon), max(lon)

    print(min_lat, max_lat)
    print(min_lon, max_lon)

    lat_range = np.arange(min_lat, max_lat, grid_threshold)
    lon_range = np.arange(min_lon, max_lon, grid_threshold)
    print(len(lat_range), len(lon_range))

    try:
        with open('cleaned_data.pkl', 'rb') as infile:
            res = cPickle.load(infile)
    except:
        res = [[[] for _ in range(len(lon_range))]
               for _ in range(len(lat_range))]
        for i in range(len(ordered_lat)):
            if not i % 10000:
                print(i)
            for x in range(len(lat_range)):
                if abs(lat_range[x] - ordered_lat[i]) > grid_threshold * sqrt(2) / 4:
                    continue
                else:
                    break
            if x >= len(lat_range):
                break
            for y in range(len(lon_range)):
                if abs(lon_range[y] - ordered_lon[i]) > grid_threshold * sqrt(2) / 4:
                    continue
                else:
                    res[x][y].append(ordered_alt[i])

        for x in range(len(lat_range)):
            for y in range(len(lon_range)):
                if res[x][y]:
                    res[x][y] = np.min(res[x][y])
                else:
                    continue

        with open('cleaned_data.pkl', 'wb') as outfile:
            cPickle.dump(res, outfile)

    ret = [[] for _ in range(3)]
    rounded = []
    for x in range(len(lat_range)):
        for y in range(len(lon_range)):
            if res[x][y]:
                _alt = round(res[x][y], 1)
                too_close = False
                for i in range(len(rounded)):
                    if _alt == rounded[i]:
                        if abs(ret[0][i] - x) + abs(ret[1][i] - y) < 80:
                            too_close = True
                            break
                if too_close:
                    continue
                ret[0].append(x)
                ret[1].append(y)
                ret[2].append(res[x][y])
                rounded.append(_alt)
    return map(np.array, ret)


def plot_dtm(x, y, z, plot=True, tri=False):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ay = fig.add_subplot(122, projection='3d')

    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(x, y, z, xi, yi, interp='linear')

    ax.set_xlim3d(0, 200)
    ax.set_ylim3d(0, 220)
    ax.set_zlim3d(-100, 100)

    ay.set_xlim3d(0, 200)
    ay.set_ylim3d(0, 220)
    ay.set_zlim3d(-100, 100)

    if tri:
        ax.plot_trisurf(x, y, z, cmap='terrain', vmin=-5, vmax=20)
        ay.plot_trisurf(x, y, z, cmap='terrain', vmin=-5, vmax=20)
        ax.scatter(x, y, z, color='white')
    else:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        linewidth=1, antialiased=True, cmap='terrain', vmin=-5, vmax=20)
        ay.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        linewidth=1, antialiased=True, cmap='terrain', vmin=-5, vmax=20)
        ax.scatter(x, y, z, color='white')

    if plot:
        plt.show()


def main():
    x, y, z = load_data('point_cloud.obj')
    x, y, z = clean_data(x, y, z)
    print(list(map(len, (x, y, z))))
    plot_dtm(x, y, z)

if __name__ == "__main__":
    main()
