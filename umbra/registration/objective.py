from typing import Tuple
import numpy as np
import cv2

from umbra.common import transform

def correlation(img1, img2):
    img1 = cv2.dft(img1, flags=cv2.DFT_COMPLEX_OUTPUT)
    img2 = cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT)
    correlation_spectrum = cv2.mulSpectrums(img1, img2, 0, conjB=True)
    return cv2.idft(correlation_spectrum, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

def correlation_peak(img1, img2) -> Tuple[int, int]:
    '''
    The highest peak of the correlation between img1 and img2 minimizes the MSE w.r.t. integer translation img2 -> img1.
    We typically use it as an initial guess for the rigid registration.
    '''
    correlation_img = correlation(img1, img2)
    ty, tx = np.unravel_index(np.argmax(correlation_img), correlation_img.shape)
    # Convert tx, ty to proper range
    h, w = img1.shape[0:2]
    ty = ty if ty <= h // 2 else ty - h # ty in [0,h-1] -> [-h//2+1, h//2]
    tx = tx if tx <= w // 2 else tx - w
    return tx, ty
    
class RigidRegistrationObjective:
    def __init__(self, ref_img, img, rotation_center, theta_factor=180/np.pi):
        '''
        Parameters
        ----------
        - ref_img : reference image.
        - img : image to register.
        - rotation_center : center of rotation for the rigid transform. /!\ We are working here \
        with the *inverse* transform, i.e. the one that maps registered_img to img. The center of rotation is \
        therefore not the same in the forward transform.
        - theta_factor : Scaling factor for theta (radians -> degrees by default) to make it comparable to tx, ty. \
        Useful when reporting results to the user, but more importantly, it can make the optimization \
        landscape more isotropic, which helps a lot for first-order methods (e.g. gradient descent). \
        Note that this is not relevant if we are using the Hessian, as it takes the curvature into account.
        '''
        dtype = img.dtype
        if dtype not in [np.float32, np.float64]:
            raise ValueError(f"Invalid img dtype {dtype}. Must be np.float32 or np.float64.")
        self.ref_img = ref_img
        self.img = img
        self.rotation_center = rotation_center
        self.theta_factor = theta_factor 

        # Precompute constants
        self.h, self.w = self.ref_img.shape
        self.x_grid, self.y_grid = np.meshgrid(np.arange(self.w, dtype=dtype) - self.rotation_center[0],
                                               np.arange(self.h, dtype=dtype) - self.rotation_center[1]) # each (H,W)
        
        self.img_grad_xy = np.zeros((*self.img.shape, 2), dtype=dtype)
        self.img_hess_xy = np.zeros((*self.img.shape, 3), dtype=dtype) # can store a symmetric (2,2) matrix in a (3,) vector

        ddepth = cv2.CV_32F if dtype == np.float32 else cv2.CV_64F
        self.img_grad_xy[...,0] = cv2.Sobel(self.img, ddepth, 1, 0) # d_x
        self.img_grad_xy[...,1] = cv2.Sobel(self.img, ddepth, 0, 1) # d_y
        self.img_hess_xy[...,0] = cv2.Sobel(self.img, ddepth, 2, 0)  # d_xx
        self.img_hess_xy[...,1] = cv2.Sobel(self.img, ddepth, 1, 1)  # d_xy = d_yx
        self.img_hess_xy[...,2] = cv2.Sobel(self.img, ddepth, 0, 2)  # d_yy
    
    def value(self, x):
        # Get the transform "img -> registered_img" determined by theta, tx, ty
        # What we actually want to apply is its inverse "img -> registered_img"
        theta, tx, ty = self.convert_x_to_params(x) 
        tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
        registered_img = transform.warp(self.img, tform.inverse.params) # is zero-padding fine here ?
        diff = registered_img - self.ref_img
        return 1/2*cv2.mean(diff*diff)[0]

    def grad(self, x):
        # Get the transform "img -> registered_img" determined by theta, tx, ty
        # What we actually want to apply is its inverse, i.e. "img -> registered_img"
        theta, tx, ty = self.convert_x_to_params(x) 
        tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
        registered_img_grad_xy = transform.warp(self.img_grad_xy, tform.inverse.params) # (H,W,2)
    
        # We now compute the gradient of the registered image with respect to theta, tx, ty for all points (x,y)
        # which is given by registered_img_grad = tform_jac.T @ registered_img_grad_xy
        # Here, tform_jac is the (2,3) Jacobian matrix of the transform (w.r.t. its parameters theta, tx, ty) at each point (x,y)
        # a) tform_jac[:,0] = tform_jac_theta depends on (x,y), so we store it in (H,W) arrays
        tform_x_dtheta = cv2.addWeighted(self.x_grid, -np.sin(theta), # (H,W)
                                         self.y_grid, -np.cos(theta),
                                         0.0)
        tform_y_dtheta = cv2.addWeighted(self.x_grid, np.cos(theta), # (H,W)
                                         self.y_grid, -np.sin(theta),
                                         0.0)
        # registered_img_grad_theta = tform_jac_theta.T @ registered_img_grad_xy, i.e. a dot product (for every point)
        registered_img_dtheta = tform_x_dtheta*registered_img_grad_xy[...,0] + \
                                tform_y_dtheta*registered_img_grad_xy[...,1] # (H,W)
        # b) Since tform_jac[:,1:] = eye(2) for all (x,y), we have registered_img_grad_t = registered_img_grad_xy
        registered_img_dtx = registered_img_grad_xy[...,0]
        registered_img_dty = registered_img_grad_xy[...,1]

        # We are now ready to compute the gradient of the objective function (MSE) with respect to theta, tx, ty
        # Simple application of the chain rule here
        objective_grad = np.zeros(3)
        objective_grad[0] = -cv2.mean(self.ref_img*registered_img_dtheta)[0]
        objective_grad[1] = -cv2.mean(self.ref_img*registered_img_dtx)[0]
        objective_grad[2] = -cv2.mean(self.ref_img*registered_img_dty)[0]
        # Compute objective gradient with respect to x (account for scaling)
        objective_grad[0] /= self.theta_factor # dx / dthetha
        return objective_grad
    
    def hess(self, x):
        # Get the transform "img -> registered_img" determined by theta, tx, ty
        # What we actually want to apply is its inverse, i.e. "img -> registered_img"
        theta, tx, ty = self.convert_x_to_params(x) 
        tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
        registered_img_grad_xy = transform.warp(self.img_grad_xy, tform.inverse.params) # (H,W,2)
        registered_img_hess_xy = transform.warp(self.img_hess_xy, tform.inverse.params) # (H,W,3)

        # We now compute the Hessian of the registered image with respect to theta, tx, ty for all points
        # img_hess = tform_jac.T @ registered_img_hess_xy @ tform_jac + tform_hess @ img_grad_xy
        # By symmetry, only need to store the (flattened) upper triangular part, i.e.
        # the second order derivatives in the order: theta^2, theta*tx, theta*ty, tx**2, tx*ty, ty**2

        # 1) tform_jac.T @ registered_img_hess_xy @ tform_jac
        # a) tform_jac[:,0] = tform_jac_theta depends on (x,y), so we store it in (H,W) arrays
        tform_x_dtheta = cv2.addWeighted(self.x_grid, -np.sin(theta), # (H,W)
                                         self.y_grid, -np.cos(theta),
                                         0.0)
        tform_y_dtheta = cv2.addWeighted(self.x_grid, np.cos(theta), # (H,W)
                                         self.y_grid, -np.sin(theta),
                                         0.0)
        # Cross terms theta^2, theta*tx, theta*ty involve tform_jac_theta
        registered_img_dtheta2 = tform_x_dtheta*tform_x_dtheta*registered_img_hess_xy[...,0] + \
                                 2*tform_x_dtheta*tform_y_dtheta*registered_img_hess_xy[...,1] + \
                                 tform_y_dtheta*tform_y_dtheta*registered_img_hess_xy[...,1]
        registered_img_dthetatx = tform_x_dtheta*registered_img_hess_xy[...,0] + \
                                  tform_y_dtheta*registered_img_hess_xy[...,1]
        registered_img_dthetaty = tform_x_dtheta*registered_img_hess_xy[...,1] + \
                                  tform_y_dtheta*registered_img_hess_xy[...,2]
        
        # b) Since tform_jac[:,1:] = eye(2) for all (x,y), we have registered_img_hess_txty = registered_img_hess_xy
        registered_img_hess_txty = registered_img_hess_xy

        # 2) tform_hess @ registered_img_grad_xy
        # where tform_hess is the (2,3,3) Hessian tensor of the transform (w.r.t. theta, tx, ty)
        # tform_hess[:,0,0] = tform_jac_theta is the only non-zero component
        tform_x_dtheta2 = cv2.addWeighted(self.x_grid, -np.cos(theta), # (H,W)
                                          self.y_grid, np.sin(theta),
                                          0.0)
        tform_y_dtheta2 = cv2.addWeighted(self.x_grid, -np.sin(theta), # (H,W)
                                          self.y_grid, -np.cos(theta),
                                          0.0)
        registered_img_dtheta2 += tform_x_dtheta2*registered_img_grad_xy[...,0] + \
                                  tform_y_dtheta2*registered_img_grad_xy[...,1]

        # Compute objective Hessian
        objective_hess = np.zeros(6)
        objective_hess[0] = -cv2.mean(self.ref_img*registered_img_dtheta2)[0] # (6,)
        objective_hess[1] = -cv2.mean(self.ref_img*registered_img_dthetatx)[0]
        objective_hess[2] = -cv2.mean(self.ref_img*registered_img_dthetaty)[0]
        for i in range(3):
            objective_hess[i+3] = -cv2.mean(self.ref_img*registered_img_hess_txty[...,i])[0]
        # Compute objective Hessian with respect to x (account for scaling)
        objective_hess[0] /= self.theta_factor**2 # chain rule -> multiply by (dx / dthetha)^2
        objective_hess[1] /= self.theta_factor
        objective_hess[2] /= self.theta_factor
        return np.array([[objective_hess[0], objective_hess[1], objective_hess[2]],
                         [objective_hess[1], objective_hess[3], objective_hess[4]],
                         [objective_hess[2], objective_hess[4], objective_hess[5]]])

    
    def convert_x_to_params(self, x):
        theta, tx, ty = x[0]/self.theta_factor, x[1], x[2] # parameters of registered_img -> img (we call it the *inverse* transform)
        return theta, tx, ty
      
    def convert_params_to_x(self, theta, tx, ty):
        x = np.array([theta*self.theta_factor, tx, ty])
        return x
    
    ### Slower grad and hess methods that use numpy instead of OpenCV
    # def grad(self, x):
    #     # Get the transform "img -> registered_img" determined by theta, tx, ty
    #     # What we actually want to apply is its inverse, i.e. "img -> registered_img"
    #     theta, tx, ty = self.convert_x_to_params(x) 
    #     tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
    #     registered_img_grad_xy = transform.warp(self.img_grad_xy, tform.inverse.params) # (H,W,2)

    #     tform_x_dtheta = - self.x_grid*np.sin(theta) - self.y_grid*np.cos(theta)
    #     tform_y_dtheta = + self.x_grid*np.cos(theta) - self.y_grid*np.sin(theta)

    #     registered_img_dtheta = tform_x_dtheta*registered_img_grad_xy[...,0] + \
    #                             tform_y_dtheta*registered_img_grad_xy[...,1] # (H,W)
        
    #     registered_img_dtx = registered_img_grad_xy[...,0]
    #     registered_img_dty = registered_img_grad_xy[...,1]

    #     objective_grad = np.zeros(3)
    #     objective_grad[0] = -np.mean(self.ref_img*registered_img_dtheta)
    #     objective_grad[1] = -np.mean(self.ref_img*registered_img_dtx)
    #     objective_grad[2] = -np.mean(self.ref_img*registered_img_dty)
    #     # Compute objective gradient with respect to x (account for scaling)
    #     objective_grad[0] /= self.theta_factor # dx / dthetha
    #     return objective_grad
    
    # def hess(self, x):
    #     # Get the transform "img -> registered_img" determined by theta, tx, ty
    #     # What we actually want to apply is its inverse, i.e. "img -> registered_img"
    #     theta, tx, ty = self.convert_x_to_params(x) 
    #     tform = transform.centered_rigid_transform(center=self.rotation_center, rotation=theta, translation=(tx, ty)) 
    #     registered_img_grad_xy = transform.warp(self.img_grad_xy, tform.inverse.params) # (H,W,2)
    #     registered_img_hess_xy = transform.warp(self.img_hess_xy, tform.inverse.params) # (H,W,3)

    #     # We now compute the Hessian of the registered image with respect to theta, tx, ty for all points
    #     # img_hess = tform_jac.T @ registered_img_hess_xy @ tform_jac + tform_hess @ img_grad_xy
    #     # By symmetry, only need to store the (flattened) upper triangular part, i.e.
    #     # the second order derivatives in the order: theta^2, theta*tx, theta*ty, tx**2, tx*ty, ty**2

    #     # 1) tform_jac.T @ registered_img_hess_xy @ tform_jac
    #     # a) tform_jac[:,0] = tform_jac_theta depends on (x,y), so we store it in (H,W) arrays
    #     tform_x_dtheta = - self.x_grid*np.sin(theta) - self.y_grid*np.cos(theta)
    #     tform_y_dtheta = + self.x_grid*np.cos(theta) - self.y_grid*np.sin(theta)
    #     # Cross terms theta^2, theta*tx, theta*ty involve tform_jac_theta
    #     registered_img_dtheta2 = tform_x_dtheta*tform_x_dtheta*registered_img_hess_xy[...,0] + \
    #                              2*tform_x_dtheta*tform_y_dtheta*registered_img_hess_xy[...,1] + \
    #                              tform_y_dtheta*tform_y_dtheta*registered_img_hess_xy[...,1]
    #     registered_img_dthetatx = tform_x_dtheta*registered_img_hess_xy[...,0] + \
    #                               tform_y_dtheta*registered_img_hess_xy[...,1]
    #     registered_img_dthetaty = tform_x_dtheta*registered_img_hess_xy[...,1] + \
    #                               tform_y_dtheta*registered_img_hess_xy[...,2]
    #     # b) Since tform_jac[:,1:] = eye(2) for all (x,y), we have registered_img_hess_txty = registered_img_hess_xy
    #     registered_img_hess_txty = registered_img_hess_xy

    #     # 2) tform_hess @ registered_img_grad_xy
    #     # where tform_hess is the (2,3,3) Hessian tensor of the transform (w.r.t. theta, tx, ty)
    #     # tform_hess[:,0,0] = tform_jac_theta is the only non-zero component
    #     tform_x_dtheta2 = - self.x_grid*np.cos(theta) + self.y_grid*np.sin(theta) # (H,W)
    #     tform_y_dtheta2 = - self.x_grid*np.sin(theta) - self.y_grid*np.cos(theta)

    #     registered_img_dtheta2 += tform_x_dtheta2*registered_img_grad_xy[...,0] + \
    #                               tform_y_dtheta2*registered_img_grad_xy[...,1]

    #     # Compute objective Hessian
    #     objective_hess = np.zeros(6)
    #     objective_hess[0] = -np.mean(self.ref_img*registered_img_dtheta2) # (6,)
    #     objective_hess[1] = -np.mean(self.ref_img*registered_img_dthetatx)
    #     objective_hess[2] = -np.mean(self.ref_img*registered_img_dthetaty)
    #     for i in range(3):
    #         objective_hess[i+3] = -np.mean(self.ref_img*registered_img_hess_txty[...,i])
    #     # Compute objective Hessian with respect to x (account for scaling)
    #     objective_hess[0] /= self.theta_factor**2 # chain rule -> multiply by (dx / dthetha)^2
    #     objective_hess[1] /= self.theta_factor
    #     objective_hess[2] /= self.theta_factor
    #     return np.array([[objective_hess[0], objective_hess[1], objective_hess[2]],
    #                      [objective_hess[1], objective_hess[3], objective_hess[4]],
    #                      [objective_hess[2], objective_hess[4], objective_hess[5]]])