<div style="display: flex; justify-content: center;">

  <div style="text-align: center;">
    <img src="docs/assets/moon_registration.gif" alt="Animated moon registration demo" width="90%"/>
    <br/>
    <em>Moon registration</em>
  </div>

  <div style="text-align: center;">
    <img src="docs/assets/sun_registration.gif" alt="Animated sun registration demo" width="90%"/>
    <br/>
    <em>Sun registration</em>
  </div>

</div>

# Install
To use Umbra, download the appropriate `.zip` file for your operating system from the [latest release](https://github.com/LudoRey/umbra/releases/latest). Inside the extracted folder, you will find the executable. **Do not move the executable out of the folder**. If you want to have a standalone file on your desktop, create a shortcut.

Note: some antivirus software may flag the executable as a false positive; if this happens, add it to your antivirus exclusion list.

# Quick start

The software should be self-explanatory : you can hover over almost any element to see a tooltip explaining its function. To make things even easier, this quick guide will help you get started.

## Registration

Before using Umbra for image registration, you must convert your (debayered) images to FITS format, and place them in a single folder. **Do not organize your images into subfolders**.


1. Open the folder containing your FITS images.
<img src="docs/assets/quick_start/open_folder.gif" width="100%">
2. Select a reference <img src="docs/assets/quick_start/target.png" style="height:0.75em;"> and an anchor <img src="docs/assets/quick_start/anchor.png" style="height:0.75em;">. Hover over the fileviewer header icons to see tips on how to make your selection.
<img src="docs/assets/quick_start/select_reference_and_anchor.gif" width="100%">
3. Set your image scale and click on start.
<img src="docs/assets/quick_start/set_image_scale_and_run.gif" width="100%">

The registered images are saved in the folders specified by the options "Moon-registered folder" and "Sun-registered folder". By default, they are located in the parent folder of the input folder.

# CLI (Advanced)

The source code for the algorithms is available if you want to run the scripts without using the GUI.

## Requirements

- Python 3.7+
- Packages in `requirements.txt`

## Registration

The script `scripts/registration.py` simultaneously performs a moon-based and a sun-based registration of the input images.

Parameters (defined at the bottom of the script) :
- `input_dir` : Input folder containing the FITS files.
- `ref_filename` : All images will be both moon-aligned and sun-aligned to this <b>reference image</b>. Ideally, the reference image should be a long exposure with the inner solar corona clipped. It should have the same camera settings as the anchor images.
- `anchor_filenames` : The <b>anchor images</b> are the only images that will be explicitly sun-aligned to the reference using the sun registration algorithm. The other images will use timestamp-based interpolation to compute the relative translation of the sun (with respect to the moon), as well as the rotation. One or two anchor images are generally sufficient. They should have the same camera settings as the reference image, and be spaced as far apart in time as possible from it.
- `moon_registered_dir` : Output folder for the moon-registered images.
- `sun_registered_dir` : Output folder for the sun-registered images.
- `image_scale` : Resolution in arcseconds/pixel.
- `clipped_factor` : In order to easily detect the moon's border, the bright pixels surrounding the moon are clipped first, if they are not already. This parameter determines <b>the number of clipped pixels</b>. Increase to make the moon's border more defined. Decrease to prevent noise amplification (which may interfere with the edge detection algorithm). The number of clipped pixels is computed as the area of an annulus around the moon, where the outer radius is given by the moon radius, multiplied by the clipped factor.
- `edge_factor` : The moon detection algorithm works by fitting a circle to the edge of the moon. This parameter determines <b>the number of detected edge pixels</b>, displayed in red and green. Increase if a large portion of the moon's border is not detected. Decrease if other parts of the image are incorrectly detected. The number of edge pixels is given by the circonference of the moon, multiplied by the edge factor. Some edge pixels are then discarded, due to non-maximum suppression.
- `sigma_highpass_tangential` : The sun registration algorithm works on filtered images that enhance the coronal details. This parameter defines <b>the standard deviation of the tangential high-pass filter</b>, given in degrees. A lower value emphasizes finer structures, while a higher value is more robust to noise.
- `max_iter` : Maximum number of iterations for the optimization loop. The loop will terminate early if the parameters of the alignment transform converge.

<!-- 
## Integration

The scripts `main_sun_integration.py` and `main_moon_integration.py` integrate the previously registered images located in `MOON_REGISTERED_DIR` and `SUN_REGISTERED_DIR`. A stack is generated for each group (see `GROUP_KEYWORDS`). The output directories are defined by `MOON_STACKS_DIR` and `SUN_STACKS_DIR`.

The `main_sun_integration.py` script performs a weighted average of each pixel in order to reject as many moon pixels as possible. For each sub, a moon mask is computed, which depends on two additional parameters : 
- `EXTRA_RADIUS_PIXELS` : extra amount of pixels added to the radius of the moon mask (which is obtained from `MOON_RADIUS_DEGREE` and `IMAGE_SCALE`). Increasing this parameter will lead to fewer artifacts at the cost of worse SNR : it should be as close to 0 as possible.
- `SMOOTHNESS` : smoothness of the mask in pixels. Increasing this parameter leads to a smoother transition at the cost of worse SNR.

## HDR composition

The scripts `main_sun_hdr_composition.py` and `main_moon_hdr_composition.py` combine the previously generated stacks located in `MOON_STACKS_DIR` and `SUN_STACKS_DIR`. The output directories are defined by `MOON_HDR_DIR` and `SUN_HDR_DIR`.

Because they are stored in 16-bit files, the pixel values of an image taken with a 14-bit sensor typically saturate at 0.25 (in the normalized [0,1] range), but this value can even be lower based on the full well capacity (FWC) of the sensor. Even then, the sensor might not be linear near the saturation point : values above ~80-90% of the saturation point are often not representative of the true brightness. Similarly, values that are near 0 suffer from the same issues. In order to create a smooth and realistic HDR composite, those too-bright and too-dark values should be rejected by the HDR algorithm. However, those thresholds uniquely depend on the imaging system, and should be derived from the images themselves. Be careful: image calibration (bias subtraction and flat division) has a non-uniform effect on those thresholds : some pixels might saturate at a lower/higher point than others for example. It is usually best to reject more pixels than necessary (as opposed to not enough).

In essence, the HDR algorithm performs a weighted combination, where the pixels that are too bright (or too dark) are rejected based on a weighting function defined by 4 parameters :
- `HIGH_CLIPPING_THRESHOLD`, `HIGH_CLIPPING_SMOOTHNESS` : values in [0,1]. The weight function is equal to 1 for pixel values below `HIGH_CLIPPING_THRESHOLD`, and equal to 0 above `HIGH_CLIPPING_THRESHOLD`+`HIGH_CLIPPING_SMOOTHNESS`. Between the two, it is a simple linear interpolation. 
- `LOW_CLIPPING_THRESHOLD`, `LOW_CLIPPING_SMOOTHNESS` : analogous to `HIGH_CLIPPING_THRESHOLD` and `HIGH_CLIPPING_SMOOTHNESS`.

Moreover, `main_sun_hdr_composition.py` uses a fitting routine before combining the images. The fit is computed on a region of appropriate brightness (as defined by `HIGH_CLIPPING_THRESHOLD` and `LOW_CLIPPING_THRESHOLD`), which also excludes the moon. Similarly to `main_sun_integration.py`, the script uses an additional parameter for the moon mask :
- `EXTRA_RADIUS_PIXELS` : extra amount of pixels added to the radius of the moon mask (which is obtained from `MOON_RADIUS_DEGREE` and `IMAGE_SCALE`).

## Moon and sun composition 

The script `main_merge_sun_moon.py` combines the previously generated HDR images located in `MOON_HDR_DIR` and `SUN_HDR_DIR`. The output directory is defined by `MERGED_HDR_DIR`. 

The script uses a moon mask (once again!), but this time it is not approximated by a disk but rather directly estimated from the image. 
- `MOON_THRESHOLD` : value in [0,1]. Only moon pixels below this value will be considered for the initial moon mask. This value should be increased to contain more of the moon edge, but it should not be too high (to avoid artifacts). 
- `SIGMA` : value above 0. Roughly corresponds to "outwards-only" Gaussian smoothing (but there is more to it, more explanations will come later). -->
