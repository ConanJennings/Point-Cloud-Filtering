from math import sin, cos, sqrt
import numpy as np


def lla2xyz(lat, lon, alt):
    lat = np.radians(lat)
    lon = np.radians(lon)

    a = 6378137.0
    f = 1.0 / 298.257224
    e = 8.1819190842622e-2
    asq = a * a
    esq = e * e
    N = a / sqrt(1 - esq * sin(lat) ** 2)

    C = 1.0 / sqrt(cos(lat)**2 + (1.0 - f) ** 2 * sin(lat) ** 2)
    S = (1.0 - f) ** 2 * C

    x = (a * C + alt) * cos(lat) * cos(lon)
    y = (a * C + alt) * cos(lat) * sin(lon)
    z = (a * S + alt) * sin(lat)

    ecef = np.array([x, y, z])
    ref = np.array([4363888.148175386, 850492.0731233275, 4557825.17188836])
    R = [[-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
         [-sin(lon), cos(lon), 0.0],
         [-cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat)]]
    R = np.array(R)
    ned = np.dot(R.T, ecef - ref)
    z, y, x = ned
    # return x, y, z
    return x, y, -z


def convert_points(lat, lon, alt, intensity):
    print(np.median(lat), np.median(lon))
    x, y, z = [], [], []
    outfile = open('point_cloud.obj', 'w')
    outfile.write('mtllib ./vp.mtl\n\ng\n')
    for i in range(len(lat)):
        X, Y, Z = lla2xyz(lat[i], lon[i], alt[i])
        outfile.write(
        ' '.join(map(str, ['v', X, Y, Z])) + '\n')
        x.append(X)
        y.append(Y)
        z.append(Z)
    outfile.close()
    print('finished conversion')
    return map(np.array, [x, y, z])
