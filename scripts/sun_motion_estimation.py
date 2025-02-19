if __name__ == "__main__":
    from _import import add_project_root_to_path
    add_project_root_to_path()

import os

from matplotlib import pyplot as plt
import numpy as np
import skimage as sk
import cv2
from matplotlib.animation import FuncAnimation

#from core.lib.registration import sun, optim, transform, utils
from core.lib import fits, display, registration, transform, optim
from core.lib.utils import cprint

def main(input_dir,
        ref_filename,
        other_filename,
        *,
        img_callback=lambda img: None,
        checkstate=lambda: None):

    # Load images
    ref_img, ref_header = fits.read_fits_as_float(os.path.join(input_dir, ref_filename))
    img, header = fits.read_fits_as_float(os.path.join(input_dir, other_filename))

    # Get clipping value
    print("Computing clipping value...", end=" ", flush=True)
    clipping_value = min(registration.sun.get_clipping_value(img, header) for img, header in zip([ref_img, img], [ref_header, header]))
    print(f"{clipping_value:.3f}")

    # Preprocess reference image
    ref_img = registration.sun.preprocess(ref_img, ref_header, clipping_value)
    # GUI interactions
    checkstate()
    img_to_display = red_cyan_colormap(ref_img, np.median(ref_img))
    img_callback(img_to_display)

    # Preprocess image to register
    img = registration.sun.preprocess(img, header, clipping_value)
    # GUI interactions
    checkstate()
    img_to_display = red_cyan_colormap(ref_img, img)
    img_callback(img_to_display)

    tx, ty = registration.correlation_peak(img, ref_img) # translation ref_img -> img
    theta = 0
    rotation_center = (ref_header["MOON-X"], ref_header["MOON-Y"])

    def optim_callback(iter, x, delta, f, g):
        # Checkstate
        checkstate()
        # Display info
        print(f"\nIteration {iter}:")
        cprint(f"theta: {x[0]:>8.3f} ({delta[0]:+.3f})", color="green") 
        cprint(f"tx   : {x[1]:>8.2f} ({delta[1]:+.2f})", color="green") 
        cprint(f"ty   : {x[2]:>8.2f} ({delta[2]:+.2f})", color="green") 
        print(f"Objective value   : {f:.3e}")
        print(f"Objective gradient: {g[0]:.3e}, {g[1]:.3e}, {g[2]:.3e}")
        # Custom image callback
        theta, tx, ty = obj.convert_x_to_params(x)
        inv_tform = transform.centered_rigid_transform(center=rotation_center, rotation=theta, translation=(tx,ty))
        img_to_display = red_cyan_colormap(ref_img, transform.warp(img, inv_tform.inverse.params))
        
        img_callback(img_to_display)

    obj = registration.RigidRegistrationObjective(ref_img, img, rotation_center, theta_factor=180/np.pi)
    delta_max = 1
    delta_min = np.array([1e-4, 1e-3, 1e-3]) # want more precision on the angle
    x = optim.line_search_newton(obj.convert_params_to_x(theta, tx, ty),
                                 obj.value, obj.grad, obj.hess,
                                 delta_max=delta_max, delta_min=delta_min,
                                 callback=optim_callback)
    theta, tx, ty = obj.convert_x_to_params(x)

def red_cyan_colormap(img1, img2, difference_blend_factor=0.75):
    color_img = np.zeros((*img1.shape, 3), dtype=np.float64)
    mean = (img1 + img2)/2
    diff = img1 - img2
    color_img[..., 0] = (1-difference_blend_factor)*mean + difference_blend_factor*diff
    color_img[..., 1] = (1-difference_blend_factor)*mean - difference_blend_factor*diff
    color_img[..., 2] = color_img[..., 1]
    return display.normalize(color_img)

if __name__ == "__main__":
    import sys
    from core.lib.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    from core.parameters import IMAGE_SCALE
    from core.parameters import MEASURED_TIME, UTC_TIME, LATITUDE, LONGITUDE
    from core.parameters import INPUT_DIR

    ref_filename = "0.25000s_2024-04-09_02h40m33s.fits"
    other_filename = "0.25000s_2024-04-09_02h42m31s.fits"

    # ref_filename = "0.00100s_2024-04-09_02h39m59s.fits"
    # other_filenames = ["0.00100s_2024-04-09_02h43m02s.fits"]

    main(INPUT_DIR,
         ref_filename,
         other_filename)
    
    ### Legacy debug code below

    # fig0, ax0 = plt.subplots()
    # ax0.imshow(gaussian_filter((ref_img - registered_img)**2, sigma=10))

    # fig1, ax1 = plt.subplots()
    # anim_img = ax1.imshow(ref_img, animated=True)

    # def update(frame):
    #     if frame % 2 == 0:
    #         anim_img.set_data(ref_img)
    #     else:
    #         anim_img.set_data(registered_img)
    #     return [anim_img]

    # ani = FuncAnimation(fig1, update, frames=30, interval=500, blit=True)

    # # Cross correlation image
    # fig2, ax2 = plt.subplots()
    # # Centering
    # h, w = correlation_img.shape
    # extent = [-w//2, w//2-1, h//2 - 1, -h//2]
    # ax2.imshow(np.fft.fftshift(correlation_img), extent=extent)
    # # Zoom
    # r = 4
    # rolled_img = np.roll(np.roll(correlation_img, -ty+r, axis=0), -tx+r, axis=1)[:2*r+1, :2*r+1]
    # ax.imshow(rolled_img, extent=[tx-r, tx+r, ty+r, ty-r])

    # # Grid search registration
    # num = 5
    # theta_range = np.linspace(-np.deg2rad(0.062), -np.deg2rad(0.067), num)
    # tx_range = np.linspace(tx+0.16, tx+0.21, num)
    # ty_range = np.linspace(ty+0.26, ty+0.31, num)

    # values = np.zeros((num, num, num))
    # for i, theta in enumerate(theta_range):
    #     for j, tx in enumerate(tx_range):
    #         for k, ty in enumerate(ty_range):
    #             values[i,j,k] = obj.value(obj.convert_params_to_x(theta, tx, ty))

    # fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})

    # vmin, vmax = values.min(), values.max()
    # X, Y = np.meshgrid(tx_range, ty_range, indexing='ij')

    # def update_surf(frame):
    #     theta_idx = frame % num
    #     tx_idx, ty_idx = np.unravel_index(np.argmin(values[theta_idx]), values[theta_idx].shape)
    #     tx, ty = tx_range[tx_idx], ty_range[ty_idx]
    #     value = values[theta_idx, tx_idx, ty_idx]
    #     ax3.cla()
    #     ax3.plot_surface(X, Y, values[theta_idx], edgecolor='0.7', alpha=0.8)
    #     ax3.scatter(tx, ty, value, c='red', label=f"theta: {np.rad2deg(theta_range[theta_idx]):.3f} \ntx: {tx:.2f} \nty: {ty:.2f} \nvalue: {value:.2e}")
    #     ax3.set_zlim(vmin, vmax)
    #     ax3.legend()

    #     return [fig3]
    
    # update_surf(0)
    # surf_ani = FuncAnimation(fig3, update_surf, frames=30, interval=500)
    
    # plt.show()