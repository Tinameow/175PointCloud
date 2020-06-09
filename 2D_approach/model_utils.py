import numpy as np
import math
import random
import os
from torch.utils.data import Dataset
import torch
from path import Path
from torchvision import transforms, utils

path = Path("../Data/ModelNet10")
folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


def makerotation(rx, ry, rz):
    """
    Generate a rotation matrix

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    rx = np.pi * rx / 180.0
    ry = np.pi * ry / 180.0
    rz = np.pi * rz / 180.0

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = (Rz @ Ry @ Rx)

    return R


class Camera:
    """
    A simple data structure describing camera parameters

    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation


    """

    def __init__(self, f, c, R, t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'

    def project(self, pts3):
        """
        Project the given 3D points in world coordinates into the specified camera

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert (pts3.shape[0] == 3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)

        # project
        p = self.f * (pcam / pcam[2, :])

        # offset principal point
        pts2 = p[0:2, :] + self.c

        assert (pts2.shape[1] == pts3.shape[1])
        assert (pts2.shape[0] == 2)

        return pts2

    def update_extrinsics(self, params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.

        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation

        """
        self.R = makerotation(params[0], params[1], params[2])
        self.t = np.array([[params[3]], [params[4]], [params[5]]])


class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class create_data_point(object):
    def __init__(self, cams):
        self.cams = cams

    def __call__(self, pointcloud):
        n = len(self.cams)
        pts2 = np.zeros((n, 32, 32))
        for i in range(n):
            ind = self.cams[i].project(pointcloud.T).astype(int).T
            for k, j in ind:
                if 0 <= k < 32 and 0 <= j < 32:
                    pts2[i, j, k] += 1
        pts2 /= np.max(pts2)
        return pts2


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}


class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def get_cams(c: str):
    if c == "4cams":
        cam1 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 2, 0]]).T, R=makerotation(90, 0, 0))
        cam2 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[2, 0, 0]]).T,
                      R=makerotation(0, 90, 0) @ makerotation(0, 0, 270))
        cam3 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 0, 2]]).T, R=makerotation(180, 0, 0))
        cam4 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[1.4, 1.4, 1.4]]).T, R=makerotation(-45, -45, 0))
        # cam4 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[1.4, 1.4, 1.4]]).T, R=makerotation(135, 0, -45))
        cams = [cam1, cam2, cam3, cam4]
        return cams

    if c == "4camsup":
        cam1 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 1.4, 1.4]]).T, R=makerotation(135, 0, 0))
        cam2 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[1.4, 0, 1.4]]).T,
                      R=makerotation(0, 135, 0) @ makerotation(0, 0, 270))
        cam3 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[-1.4, 0, 1.4]]).T,
                      R=makerotation(0, 225, 0) @ makerotation(0, 0, 90))
        cam4 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, -1.4, 1.4]]).T,
                      R=makerotation(225, 0, 0) @ makerotation(0, 0, 180))
        cams = [cam1, cam2, cam3, cam4]
        return cams

    if c == "6cams":
        cam1 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 2, 0]]).T, R=makerotation(90, 0, 0))
        cam2 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[2, 0, 0]]).T,
                      R=makerotation(0, 90, 0) @ makerotation(0, 0, 270))
        cam3 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 0, 2]]).T, R=makerotation(180, 0, 0))
        cam4 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[1.4, 1.4, 0]]).T, R=makerotation(90, 0, -45))
        cam5 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 1.4, 1.4]]).T, R=makerotation(-225, 0, 0))
        cam6 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[1.4, 0, 1.4]]).T,
                      R=makerotation(0, -225, 0) @ makerotation(0, 0, 270))
        cams = [cam1, cam2, cam3, cam4, cam5, cam6]
        return cams

    if c == "3cams":
        cam1 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 2, 0]]).T, R=makerotation(90, 0, 0))
        cam2 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[2, 0, 0]]).T,
                      R=makerotation(0, 90, 0) @ makerotation(0, 0, 270))
        cam3 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 0, 2]]).T, R=makerotation(180, 0, 0))
        cams = [cam1, cam2, cam3]
        return cams