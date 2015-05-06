import numpy as np
import math
import yaml
from scipy.special import cbrt
import cv2

# Bibliography:
# [1] Sara R. Matousek M. 3D Computer Vision. January 7, 2014.
#     Online: http://cmp.felk.cvut.cz/cmp/courses/TDV/2013W/lectures/tdv-2013-all.pdf


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.
    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)
    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert(type(projective) == np.ndarray)
    assert((projective.shape[0] == 4) | (projective.shape[0] == 3))
    return (projective / projective[-1, :])[0:-1, :]


def e2p(euclidean):
    """
    Convert 2d or 3d euclidean to projective coordinates.
    :param euclidean: projective coordinate(s)
    :type euclidean: numpy.ndarray, shape=(2 or 3, n)
    :return: projective coordinate(s)
    :rtype: numpy.ndarray, shape=(3 or 4, n)
    """
    assert(type(euclidean) == np.ndarray)
    assert((euclidean.shape[0] == 3) | (euclidean.shape[0] == 2))
    return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


class Camera:
    """
    Projective camera model
        - camera intrinsic and extrinsic parameters handling
        - various lens distortion models
        - model persistence
        - projection of camera coordinates to an image
        - conversion of image coordinates on a plane to camera coordinates
        - visibility handling
    """
    def __init__(self, id=None):
        """
        :param id: camera identification number
        :type id: unknown or int
        """
        self.K = np.eye(3)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.update_P()
        self.kappa = np.zeros((2,))
        self.id = id
        self.size_px = np.zeros((2,))

        self.bouguet_kc = np.zeros((5,))
        self.kannala_p = np.zeros((6,))
        self.kannala_thetamax = None
        self.calibration_type = 'standard'  # other possible values: bouguet, kannala

    def save(self, filename):
        """
        Save camera model to a YAML file.
        """
        data = {'id': self.id,
                'K': self.K.tolist(),
                'R': self.R.tolist(),
                't': self.t.tolist(),
                'size_px': self.size_px.tolist(),
                'calibration_type': self.calibration_type
                }
        if self.calibration_type == 'bouguet':
            data['bouguet_kc'] = self.bouguet_kc.tolist()
        elif self.calibration_type == 'kannala':
            data['kannala_p'] = self.kannala_p.tolist()
            data['kannala_thetamax'] = self.kannala_thetamax
        elif self.calibration_type == 'tsai':
            data_tsai = {'tsai_f': self.tsai_f,
                         'tsai_kappa': self.tsai_kappa,
                         'tsai_nfx': self.tsai_nfx,
                         'tsai_dx': self.tsai_dx,
                         'tsai_dy': self.tsai_dy,
                         'tsai_ncx': self.tsai_ncx,
                         }
            data.update(data_tsai)
        else:
            data['kappa'] = self.kappa.tolist()
        yaml.dump(data, open(filename, 'w'))

    def load(self, filename):
        """
        Load camera model from a YAML file.
        Example::
            calibration_type: standard
            K:
            - [1225.2, -7.502186291576686e-14, 480.0]
            - [0.0, 1225.2, 384.0]
            - [0.0, 0.0, 1.0]
            R:
            - [-0.9316877145365, -0.3608289515885, 0.002545329627547]
            - [-0.1725273110187, 0.4247524018287, -0.8888909933995]
            - [0.3296724908378, -0.8263880720441, -0.4579894432589]
            id: 0
            kappa: [0.0, 0.0]
            size_px: [960, 768]
            t:
            - [-1.365061486465]
            - [3.431608806127]
            - [17.74182159488]
        """
        data = yaml.load(open(filename))
        self.id = data['id']
        self.K = np.array(data['K']).reshape((3, 3))
        self.R = np.array(data['R']).reshape((3, 3))
        self.t = np.array(data['t']).reshape((3, 1))
        self.size_px = np.array(data['size_px']).reshape((2,))
        self.calibration_type = data['calibration_type']
        if self.calibration_type == 'bouguet':
            self.bouguet_kc = np.array(data['bouguet_kc']).reshape((5,))
        elif self.calibration_type == 'kannala':
            self.kannala_p = np.array(data['kannala_p']).reshape((6,))
            self.kannala_thetamax = data['kannala_thetamax']  # not used now
            # Focal length actually used is from kannala_p. Why then K is stored? Works for me like this.
            self.K[0, 0] = self.kannala_p[2]
            self.K[1, 1] = self.kannala_p[3]
            # principal point in K and kannala_p[4:] should be consistent
            assert self.K[0, 2] == self.kannala_p[4]
            assert self.K[1, 2] == self.kannala_p[5]
        elif self.calibration_type == 'tsai':
            self.tsai_f = data['tsai_f']
            self.tsai_kappa = data['tsai_kappa']
            self.tsai_ncx = data['tsai_ncx']
            self.tsai_nfx = data['tsai_nfx']
            self.tsai_dx = data['tsai_dx']
            self.tsai_dy = data['tsai_dy']
        else:
            self.kappa = np.array(data['kappa']).reshape((2,))
        self.update_P()

    def update_P(self):
        """
        Update camera P matrix from K, R and t.
        """
        self.P = self.K.dot(np.hstack((self.R, self.t)))

    def set_K(self, K):
        """
        Set K and update P.
        :param K: intrinsic camera parameters
        :type K: numpy.ndarray, shape=(3, 3)
        """
        self.K =[[1220.2, 0, 320.0]
                 [0.0, 1220.2, 176.0]
                 [0.0, 0.0, 1.0]]      
        
        #self.K = K
        self.update_P()

    def set_K_elements(self, f, theta_rad, a, u0_px, v0_px):
        """
        Update pinhole camera intrinsic parameters and updates P matrix.
        :param f: focal length
        :type f: double
        :param theta_rad: digitization raster skew (radians)
        :type theta_rad: double
        :param a: pixel aspect ratio
        :type a: double
        :param u0_px: principal point x position (pixels)
        :type u0_px: double
        :param v0_px: principal point y position (pixels)
        :type v0_px: double
        """
        self.K = np.array([[f, -f * 1 / math.tan(theta_rad), u0_px],
                           [0, f / (a * math.sin(theta_rad)), v0_px],
                           [0, 0, 1]])
        
        
        self.update_P()

    def set_R(self, R):
        """
        Set camera extrinsic parameters and updates P.
        :param R: camera extrinsic parameters matrix
        :type R: numpy.ndarray, shape=(3, 3)
        """
        
        
        
        r, trash = cv2.Rodrigues(R)
        
        r[0] = r[0] + math.radians(0)   #roll
        r[1] = r[1] + math.radians(90)#pitch
        r[2] = r[2] + math.radians(0) #yaw
        
#         print "R: " + str(math.degrees(r[0])) +\
#               " P: " + str(math.degrees(r[1])) +\
#               " Y: " + str(math.degrees(r[2]))
        
        
        self.R, trash = cv2.Rodrigues(r)
        
        
        #self.R = R
        self.update_P()

    
    
    
    def quatToRotation(self, x=0, y=0, z=0, w=0):
        matrix = np.zeros((3,3))
    
        # Repetitive calculations.
        q4_2 = w**2
        
        q12 = x * y
        q13 = x * z
        q14 = x * w
        q23 = y * z
        q24 = y * w
        q34 = z * w
    
        # The diagonal.
        matrix[0, 0] = 2.0 * (x**2 + q4_2) - 1.0
        matrix[1, 1] = 2.0 * (y**2 + q4_2) - 1.0
        matrix[2, 2] = 2.0 * (z**2 + q4_2) - 1.0
    
        # Off-diagonal.
        matrix[0, 1] = 2.0 * (q12 - q34)
        matrix[0, 2] = 2.0 * (q13 + q24)
        matrix[1, 2] = 2.0 * (q23 - q14)
    
        matrix[1, 0] = 2.0 * (q12 + q34)
        matrix[2, 0] = 2.0 * (q13 - q24)
        matrix[2, 1] = 2.0 * (q23 + q14)
        
#         r, trash = cv2.Rodrigues(matrix)
#         
#         r[2] = r[2] - math.radians(0)
#         r[1] = r[1] - math.radians(180)
        
        
        self.set_R(matrix)
    
    
    

    
    def set_R_euler_angles(self, angles):
        """
        Set rotation matrix according to euler angles and updates P.
        :param angles: 3 euler angles in radians,
        :type angles: double sequence, len=3
        """
        rx = angles[0]
        ry = angles[1]
        rz = angles[2]
        from numpy import sin
        from numpy import cos
        self.R = np.array([[cos(ry) * cos(rz),
                            cos(rz) * sin(rx) * sin(ry) - cos(rx) * sin(rz),
                            sin(rx) * sin(rz) + cos(rx) * cos(rz) * sin(ry)],
                           [cos(ry) * sin(rz),
                            sin(rx) * sin(ry) * sin(rz) + cos(rx) * cos(rz),
                            cos(rx) * sin(ry) * sin(rz) - cos(rz) * sin(rx)],
                           [-sin(ry),
                            cos(ry) * sin(rx),
                            cos(rx) * cos(ry)]
                           ])
        self.update_P()

    def set_t(self, t):
        """
        Set camera translation and updates P.
        :param t: camera translation vector
        :type t: numpy.ndarray, shape=(3, 1)
        """
        self.t = t
        self.update_P()

    def _undistort(self, distorted_image_coords):
        """
        Remove distortion from image coordinates.
        **TODO: not implemented**
        :param distorted_image_coords: real image coordinates
        :type distorted_image_coords: numpy.ndarray, shape=(2, n)
        :return: linear image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        # TODO
        return distorted_image_coords

    def _distort_bouguet(self, undistorted_centered_image_coord):
        """
        Distort centered image coordinate following Bouquet model.
        see http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        :param undistorted_centered_image_coord: linear centered image coordinate(s)
        :type undistorted_centered_image_coord: numpy.ndarray, shape=(2, n)
        :return: distorted coordinate(s)
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert undistorted_centered_image_coord.shape[0] == 2
        kc = self.bouguet_kc
        x = undistorted_centered_image_coord[0, :]
        y = undistorted_centered_image_coord[1, :]
        r_squared = x ** 2 + y ** 2

        # tangential distortion vector
        dx = np.array([2 * kc[2] * x * y + kc[3] * (r_squared + 2 * x ** 2),
                       kc[2] * (r_squared + 2 * y ** 2) + 2 * kc[3] * x * y])
        distorted = (1 + kc[0] * r_squared + kc[1] * r_squared ** 2 + kc[4] * r_squared ** 3) * \
            undistorted_centered_image_coord + dx
        return distorted

    def _distort_kannala(self, camera_coords):
        """
        Distort image coordinate following Kannala model (M6 version only)
        See http://www.ee.oulu.fi/~jkannala/calibration/calibration_v23.tar.gz :genericproj.m
        Juho Kannala, Janne Heikkila and Sami S. Brandt. Geometric camera calibration. Wiley Encyclopedia of Computer Science and Engineering, 2008, page 9.
        :param camera_coords: 3d points in camera coordinates
        :type camera_coords: numpy.ndarray, shape=(3, n)
        :return: distorted metric image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert camera_coords.shape[0] == 3
        x = camera_coords[0, :]
        y = camera_coords[1, :]
        z = camera_coords[2, :]
        k1 = self.kannala_p[0]
        k2 = self.kannala_p[1]

        # angle between ray and optical axis
        theta = np.arccos(z / np.linalg.norm(camera_coords, axis=0))

        # radial projection (Kannala 2008, eq. 17)
        r = k1 * theta + k2 * theta ** 3

        hypotenuse = np.linalg.norm(camera_coords[0:2, :], axis=0)
        hypotenuse[hypotenuse == 0] = 1  # avoid dividing by zero
        image_x = r * x / hypotenuse
        image_y = r * y / hypotenuse
        return np.vstack((image_x, image_y))

    def _undistort_tsai(self, distorted_metric_image_coord):
        """
        Undistort centered image coordinate following Tsai model.
        :param distorted_metric_image_coord: distorted METRIC image coordinates
            (metric image coordiante = image_xy * f / z)
        :type distorted_metric_image_coord: numpy.ndarray, shape=(2, n)
        :return: linear image coordinate(s)
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert distorted_metric_image_coord.shape[0] == 2
        # see http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
        x = distorted_metric_image_coord[0, :]
        y = distorted_metric_image_coord[1, :]
        r_squared = x ** 2 + y ** 2

        undistorted = (1 + self.tsai_kappa * r_squared) * distorted_metric_image_coord
        return undistorted

    def _distort_tsai(self, metric_image_coord):
        """
        Distort centered metric image coordinates following Tsai model.
        See: Devernay, Frederic, and Olivier Faugeras. "Straight lines have to be straight."
        Machine vision and applications 13.1 (2001): 14-24. Section 2.1.
        (only for illustration, the formulas didn't work for me)
        http://www.cvg.rdg.ac.uk/PETS2009/sample.zip :CameraModel.cpp:CameraModel::undistortedToDistortedSensorCoord
        Analytical inverse of the undistort_tsai() function.
        :param metric_image_coord: centered metric image coordinates
            (metric image coordiante = image_xy * f / z)
        :type metric_image_coord: numpy.ndarray, shape=(2, n)
        :return: distorted centered metric image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert metric_image_coord.shape[0] == 2
        x = metric_image_coord[0, :]  # vector
        y = metric_image_coord[1, :]  # vector
        r_u = np.sqrt(x ** 2 + y ** 2)  # vector
        c = 1.0 / self.tsai_kappa  # scalar
        d = -c * r_u  # vector

        # solve polynomial of 3rd degree for r_distorted using Cardan method:
        # https://proofwiki.org/wiki/Cardano%27s_Formula
        # r_distorted ** 3 + c * r_distorted + d = 0
        q = c / 3.  # scalar
        r = -d / 2.  # vector
        delta = q ** 3 + r ** 2  # polynomial discriminant, vector

        positive_mask = delta >= 0
        r_distorted = np.zeros((metric_image_coord.shape[1]))

        # discriminant > 0
        s = cbrt(r[positive_mask] + np.sqrt(delta[positive_mask]))
        t = cbrt(r[positive_mask] - np.sqrt(delta[positive_mask]))
        r_distorted[positive_mask] = s + t

        # discriminant < 0
        delta_sqrt = np.sqrt(-delta[~positive_mask])
        s = cbrt(np.sqrt(r[~positive_mask] ** 2 + delta_sqrt ** 2))
        # s = cbrt(np.sqrt(r[~positive_mask] ** 2 + (-delta[~positive_mask]) ** 2))
        t = 1. / 3 * np.arctan2(delta_sqrt, r[~positive_mask])
        r_distorted[~positive_mask] = -s * np.cos(t) + s * np.sqrt(3) * np.sin(t)

        return metric_image_coord * r_distorted / r_u

    def get_focal_length(self):
        """
        Get camera focal length.
        :return: focal length
        :rtype: double
        """
        return self.K[0, 0]

    def get_principal_point_px(self):
        """
        Get camera principal point.
        :return: x and y pixel coordinates
        :rtype: numpy.ndarray, shape=(1, 2)
        """
        return self.K[0:2, 2].reshape((1, 2))

    def is_visible(self, xy_px):
        """
        Check visibility of image points.
        :param xy_px: image point(s)
        :type xy_px: np.ndarray, shape=(2, n)
        :return: visibility of image points
        :rtype: numpy.ndarray, shape=(1, n), dtype=bool
        """
        assert xy_px.shape[0] == 2
        return (xy_px[0, :] >= 0) & (xy_px[1, :] >= 0) & \
               (xy_px[0, :] < self.size_px[0]) & \
               (xy_px[1, :] < self.size_px[1])

    def is_visible_world(self, world):
        """
        Check visibility of world points.
        :param world: world points
        :type world: numpy.ndarray, shape=(3, n)
        :return: visibility of world points
        :rtype: numpy.ndarray, shape=(1, n), dtype=bool
        """
        assert world.shape[0] == 3
        xy_px = p2e(self.world_to_image(world))
        return self.is_visible(xy_px)

    def get_camera_center(self):
        """
        Returns camera center in the world coordinates.
        :return: camera center in projective coordinates
        :rtype: np.ndarray, shape=(4, 1)
        """
        return self._null(self.P)

    def world_to_image(self, x, y, z):
        """
        Project world coordinates to image coordinates.
        :param world: world points in 3d projective or euclidean coordinates
        :type world: numpy.ndarray, shape=(3 or 4, n)
        :return: projective image coordinates
        :rtype: numpy.ndarray, shape=(3, n)
        """
        world = np.array([[x],
                          [y],
                          [z],
                          [1]])
        
        #assert(type(world) == np.ndarray)
        #if world.shape[0] == 3:
        #    world = e2p(world)
        camera_coords = np.hstack((self.R, self.t)).dot(world)
        if self.calibration_type == 'bouguet':
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_metric = xy / z
            image_coords_distorted_metric = self._distort_bouguet(image_coords_metric)
        elif self.calibration_type == 'tsai':
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_metric = xy * self.tsai_f / z
            image_coords_distorted_metric = self._distort_tsai(image_coords_metric)
        elif self.calibration_type == 'kannala':
            image_coords_distorted_metric = self._distort_kannala(camera_coords)
        else:
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_distorted_metric = xy / z
        return self.K.dot(e2p(image_coords_distorted_metric))

    def image_to_world(self, u, v, z=0.02):
        
        image_px = np.zeros((2,1))
        image_px[0,0] = u
        image_px[1,0] = v
        """
        Project image points with defined world z to world coordinates.
        :param image_px: image points
        :type image_px: numpy.ndarray, shape=(2 or 3, n)
        :param z: world z coordinate of the projected image points
        :type z: float
        :return: n projective world coordinates
        :rtype: numpy.ndarray, shape=(4, n)
        """
        if image_px.shape[0] == 2:
            image_px = e2p(image_px)
        image_undistorted = self._undistort(image_px)
        tmpP = np.hstack((self.P[:, [0, 1]], self.P[:, 2, np.newaxis] * z + self.P[:, 3, np.newaxis]))
        ret =  np.linalg.inv(tmpP).dot(image_undistorted)
        
        x=ret[0]/ret[2]
        y=ret[1]/ret[2]
        z=z
        w=ret[2]
        
        
        ret[0] = x
        ret[1] = y
        ret[2] = z
        #print ret
        return (ret, w)

    def plot_world_points(self, points, plot_style, label=None,
                          solve_visibility=True):
        """
        Plot world points to a matplotlib figure.
        :param points: world points (projective or euclidean)
        :type points: numpy.ndarray, shape=(3 or 4, n) or list of lists
        :param plot_style: matplotlib point and line style code, e.g. 'ro'
        :type plot_style: str
        :param label: label plotted under points mean
        :type label: str
        :param solve_visibility: if true then plot only if all points are visible
        :type solve_visibility: bool
        """
        object_label_y_shift = +25
        import matplotlib.pyplot as plt

        if type(points) == list:
            points = np.array(points)
        points = np.atleast_2d(points)
        image_points_px = p2e(self.world_to_image(points))
        if not solve_visibility or np.all(self.is_visible(image_points_px)):
            plt.plot(image_points_px[0, :],
                     image_points_px[1, :], plot_style)
            if label:
                    max_y = max(image_points_px[1, :])
                    mean_x = image_points_px[0, :].mean()
                    plt.text(mean_x, max_y + object_label_y_shift, label)

    def _null(A, eps=1e-15):
        """
        Matrix null space.
        For matrix null space holds: A * null(A) = zeros
        source: http://mail.scipy.org/pipermail/scipy-user/2005-June/004650.html
        :param A: input matrix
        :type A: numpy.ndarray, shape=(m, n)
        :param eps: values lower than eps are considered zero
        :type eps: double
        :return: null space of the matrix A
        :rtype: numpy.ndarray, shape=(n, 1)
        """
        u, s, vh = np.linalg.svd(A)
        n = A.shape[1]   # the number of columns of A
        if len(s) < n:
            expanded_s = np.zeros(n, dtype=s.dtype)
            expanded_s[:len(s)] = s
            s = expanded_s
        null_mask = (s <= eps)
        null_space = np.compress(null_mask, vh, axis=0)
        return np.transpose(null_space)


def nview_linear_triangulation(cameras, correspondences):
    """
    Computes ONE world coordinate from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param correspondences: image coordinates correspondences in n views
    :type correspondences: numpy.ndarray, shape=(2, n)
    :return: world coordinate
    :rtype: numpy.ndarray, shape=(3, 1)
    """
    assert(len(cameras) >= 2)
    assert(type(cameras) == list)
    assert(correspondences.shape == (2, len(cameras)))

    def _construct_D_block(P, uv):
        """
        Constructs 2 rows block of matrix D.
        See [1, p. 88, The Triangulation Problem]
        :param P: camera matrix
        :type P: numpy.ndarray, shape=(3, 4)
        :param uv: image point coordinates (xy)
        :type uv: numpy.ndarray, shape=(2,)
        :return: block of matrix D
        :rtype: numpy.ndarray, shape=(2, 4)
        """
        return np.vstack((uv[0] * P[2, :] - P[0, :],
                          uv[1] * P[2, :] - P[1, :]))


    D = np.zeros((len(cameras) * 2, 4))
    for cam_idx, cam, uv in zip(xrange(len(cameras)), cameras, correspondences.T):
        D[cam_idx * 2:cam_idx * 2 + 2, :] = _construct_D_block(cam.P, uv)
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    return p2e(u[:, -1, np.newaxis])


def nview_linear_triangulations(cameras, image_points):
    """
    Computes world coordinates from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param image_points: image coordinates of m correspondences in n views
    :type image_points: sequence of m numpy.ndarray, shape=(2, n)
    :return: m world coordinates
    :rtype: numpy.ndarray, shape=(3, m)
    """
    assert(type(cameras) == list)
    assert(type(image_points) == list)
    assert(len(cameras) == image_points[0].shape[1])
    assert(image_points[0].shape[0] == 2)

    world = np.zeros((3, len(image_points)))
    for i, correspondence in enumerate(image_points):
        world[:, i] = nview_linear_triangulation(cameras, correspondence)
    return world