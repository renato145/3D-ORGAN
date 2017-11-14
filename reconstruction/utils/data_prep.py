import math
import numpy as np

def get_fractured(source, min_points=1, max_points=4, min_radius=3, max_radius=6, sphere_chance=0.75):
    fractured = source.copy()
    points = np.random.randint(min_points, max_points+1)
    idxs = np.argwhere(fractured == 1)
    centers = idxs[np.random.choice(len(idxs), points, False)]
    
    for x,y,z in centers:
        r = np.random.randint(min_radius, max_radius+1)
        xmin, xmax = max(0, x-r), x+r
        ymin, ymax = max(0, y-r), y+r
        zmin, zmax = max(0, z-r), z+r
        sphere = np.ones_like(fractured)
        sphere[xmin:xmax, ymin:ymax, zmin:zmax] = 0
        # sphere or cube
        if np.random.rand() < sphere_chance:
            idxs = np.argwhere(sphere == 0)
            idxs_remove = np.sqrt((idxs[:,0] - x)**2 + (idxs[:,1] - y)**2 + (idxs[:,2] - z)**2)
            idxs_remove = idxs[idxs_remove > r]
            sphere[idxs_remove[:,0], idxs_remove[:,1], idxs_remove[:,2]] = 1
            
        fractured *= sphere
    
    return fractured

def read_points(file):
	with open(file) as f:
		d = [l.strip().split(' ') for l in f.readlines()]
		d = [[float(n) for n in l[1:]] for l in d if l[0] == 'v']
		d = np.asarray(d, dtype=np.float32)
	
	return d

def read_point_files(files):
	out = []
	for file in files:
		out.append(read_points(file))
	
	return out

def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize), dtype=np.bool)
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    
    return vol

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    
    return points

def auto_pcl_to_volume(points, vsize):
	data_min = np.min(points)
	data_max = np.max(points)
	radius = max(abs(data_min), data_max)
	radius = math.ceil(radius*100) / 100
	vol = point_cloud_to_volume(points, vsize, radius)

	return vol