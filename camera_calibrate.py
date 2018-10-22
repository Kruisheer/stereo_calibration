#To do add pickle support for faster processing
#Forked from bvnayak/stereo_calibration
#Added support for video and saving the parameters in YAML
#For any questions on the code ask Gerard Kruisheer
import numpy as np
import cv2
import glob
import yaml
import argparse
import yaml

class StereoCalibration(object):
    def __init__(self): #, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        #self.cal_path = filepath
        self.read_images()#self.cal_path)

    def read_images(self): #, cal_path):
        print("location grids")
        cap_right = cv2.VideoCapture('clip1.mp4')
        cap_left  = cv2.VideoCapture('clip2.mp4')

        #Sync the frames as close as possible
        for i in range(1):
            ret, right_image = cap_right.read()


        for j in range(16):
            ret, left_image = cap_left.read()

        # images_right = glob.glob(cal_path + 'RIGHT/*.JPG')
        # images_left = glob.glob(cal_path + 'LEFT/*.JPG')

        i = 0
        # images_left.sort()
        # images_right.sort()

        #for i, fname in enumerate(images_right):
        for i in range(700):
        #while(cap_left.isOpened() and cap_right.isOpened()):
            if i % 20 == 0: #skip frames
                ret, images_right = cap_right.read()
                ret, images_left  = cap_left.read()

                gray_l = cv2.cvtColor(images_left, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(images_right, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                if ret_l is True:
                    rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                          (-1, -1), self.criteria)
                    self.imgpoints_l.append(corners_l)

                    # Draw and display the corners
                    # ret_l = cv2.drawChessboardCorners(img_l, (9, 6), corners_l, ret_l)
                    ret_l = cv2.drawChessboardCorners(images_left, (9, 6), corners_l, ret_l)
                    cv2.imshow("images_left", images_left)


                if ret_r is True:
                    rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                          (-1, -1), self.criteria)
                    self.imgpoints_r.append(corners_r)

                    # Draw and display the corners
                    ret_r = cv2.drawChessboardCorners(images_right, (9, 6),
                                                      corners_r, ret_r)
                    cv2.imshow("images_right", images_right)
                    #cv2.waitKey(500)

                img_shape = gray_l.shape[::-1]
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            else:
                ret, images_right  = cap_right.read()
                ret, images_left  = cap_left.read()

        cv2.destroyAllWindows()
        print("Storing Calibration parameters")

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        print("Calibrating ...")
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])
        calibration_yaml = {'M1': M1.tolist(), 'M2': M2.tolist(), 'dist1': d1.tolist(), 'dist2': d2.tolist(), 'R': R.tolist(), 'T': T.tolist(), 'E': E.tolist(), 'F': F.tolist()}
        
        #cv2.destroyAllWindows()

        with open('stereoCal.yaml', 'w') as fw:
            yaml.dump(calibration_yaml, fw)
        return camera_model

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('filepath', help='String Filepath')
    #args = parser.parse_args()
    cal_data = StereoCalibration()#args.filepath)