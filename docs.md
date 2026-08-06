Umbra Source Code Documentation
==============

# Install

Make sure you have Python 3.7 or higher installed. Then, clone the repository and navigate to it:
```
git clone https://github.com/LudoRey/umbra.git
cd umbra
```

Create a virtual environment and activate it (optional, but recommended) :
```
python -m venv venv
.\venv\Scripts\activate # On Windows
source venv/bin/activate # On MacOS/Linux
```

Install Umbra and its dependencies:
```
pip install -e .
```

# Scripts

The main scripts are located in `umbra/scripts`. All parameters are defined in the configuration file `config.example.yaml`: you should copy this file to `config.yaml` and change the parameters as needed!

Some parameters are shared among different scripts, and are defined at the top of `config.yaml`:
- `raw_dir`, `fits_dir`, `moon_registered_dir`, `sun_registered_dir`, `stacks_dir` : Input/output directories for the scripts.
- `image_scale` : Resolution in arcseconds/pixel.
- `group_keywords` : List of FITS keywords corresponding to settings that vary across the exposures (typically, "EXPTIME" and optionally "ISOSPEED" or "GAIN" if the gain was changed). These keywords will automatically determine groups of images to be stacked together.


## Registration

The script `registration.py` simultaneously performs a moon-based and a sun-based registration of the input images located in `fits_dir`. The input images should be calibrated, debayered and converted to FITS format (16-bit unsigned integer or floating point preferably). The registration algorithm uses the following parameters:
- `ref_filename` : The <b>reference image</b> determines the position of the moon and sun in the registered images. Can be left to null for automatic selection, or set to the filename of one of the input images. To ensure good results later on, the reference image should be a short exposure image.
- `anchor_filenames` : The <b>anchor images</b> are the only images that will be aligned together directly through the sun registration algorithm. The other images will be aligned by first detecting the moon and then interpolating the relative position of the sun with respect to the moon. Can be left to null for automatic selection, or set to a list of filenames. At least two anchor images are required. They should be spread as far apart in time as possible and have the same camera settings: ideally, such that the inner solar corona is clipped.
- `clipped_factor` : In order to easily detect the moon's border, the bright pixels surrounding the moon are clipped first, if they are not already. This parameter determines <b>the number of clipped pixels</b>. Increase to make the moon's border more defined. Decrease to prevent noise amplification (which may interfere with the edge detection algorithm). The number of clipped pixels is computed as the area of an annulus around the moon, where the outer radius is given by the moon radius, multiplied by the clipped factor.
- `edge_factor` : The moon detection algorithm works by fitting a circle to the edge of the moon. This parameter determines <b>the number of detected edge pixels</b>, displayed in red and green. Increase if a large portion of the moon's border is not detected. Decrease if other parts of the image are incorrectly detected. The number of edge pixels is given by the circonference of the moon, multiplied by the edge factor. Some edge pixels are then discarded, due to non-maximum suppression.
- `sigma_highpass_tangential` : The sun registration algorithm works on filtered images that enhance the coronal details. This parameter defines <b>the standard deviation of the tangential high-pass filter</b>, given in degrees. A lower value emphasizes finer structures, while a higher value is more robust to noise.
- `max_iter` : Maximum number of iterations for the optimization loop. The loop will terminate early if the parameters of the alignment transform converge.


## Integration

The script `integration.py` integrates the previously registered images located in `moon_registered_dir` and `sun_registered_dir`, and merges the two alignments into a single image. One stack is generated for each group (see `group_keywords`) in the output directory `stacks_dir`.

Every group is stacked twice from the same exposures: once from the sun-registered images and once from the moon-registered ones. In the sun-registered images the corona is aligned but the moon drifts, so the moon has to be rejected; in the moon-registered images it is the other way around. Merging the two restores, at the reference frame's moon position, what the sun stack had to reject.

The threshold `outlier_threshold` is used by the sigma-clipping routine and is given in units of standard deviation.

When stacking the sun-registered images, the moon pixels are rejected to avoid "ghosting" artifacts. For each sub, a moon mask is computed, which depends on two parameters :
- `rejection_extra_radius` : extra amount of pixels added to the radius of the moon mask. Increasing this parameter will lead to fewer artifacts at the cost of worse SNR : it should be as close to 0 as possible.
- `rejection_smoothness` : smoothness of the mask in pixels. Increasing this parameter leads to a smoother transition at the cost of worse SNR.

The merge takes the moon-registered image inside the moon's disk and the sun-registered one outside it, with a transition annulus straddling the limb where the darker of the two is kept :
- `blend_smoothness` : half-width of that annulus in pixels, i.e. how far the transition reaches on each side of the limb. It only needs to cover the fact that the limb is not a perfectly sharp disk (blur, lunar relief, residual registration error), so a few pixels is enough : a large value blurs lunar detail outwards and the inner corona inwards.

<!--
Outdated: `sun_hdr_composition.py` and `moon_hdr_composition.py` are no longer runnable, pending a rewrite onto the merged stacks produced by `integration.py`.

## HDR composition

The scripts `sun_hdr_composition.py` and `moon_hdr_composition.py` combine the previously generated stacks located in `moon_stacks_dir` and `sun_stacks_dir`. The output directories are defined by `moon_hdr_dir` and `sun_hdr_dir`.

Because they are stored in 16-bit files, the pixel values of an image taken with a 14-bit sensor typically saturate at 0.25 (in the normalized [0,1] range), but this value can even be lower based on the full well capacity (FWC) of the sensor. Even then, the sensor might not be linear near the saturation point : values above ~80-90% of the saturation point are often not representative of the true brightness. Similarly, values that are near 0 suffer from the same issues. In order to create a smooth and realistic HDR composite, those too-bright and too-dark values should be rejected by the HDR algorithm. However, those thresholds uniquely depend on the imaging system, and should be derived from the images themselves. Be careful: image calibration (bias subtraction and flat division) has a non-uniform effect on those thresholds : some pixels might saturate at a lower/higher point than others for example. It is usually best to reject more pixels than necessary (as opposed to not enough).

In essence, the HDR algorithm performs a weighted combination, where the pixels that are too bright (or too dark) are rejected based on a weighting function defined by 4 parameters :
- `high_clipping_threshold`, `high_clipping_smoothness` : values in [0,1]. The weight function is equal to 1 for pixel values below `high_clipping_threshold`, and equal to 0 above `high_clipping_threshold`+`high_clipping_smoothness`. Between the two, it is a simple linear interpolation. 
- `low_clipping_threshold`, `low_clipping_smoothness` : analogous to `high_clipping_threshold` and `high_clipping_smoothness`.

Moreover, `sun_hdr_composition.py` uses a fitting routine before combining the images. The fit is computed on a region of appropriate brightness (as defined by `high_clipping_threshold` and `low_clipping_threshold`), which also excludes the moon. The script uses an additional parameter for the moon mask :
- `extra_radius_pixels` : extra amount of pixels added to the radius of the moon mask.
-->

