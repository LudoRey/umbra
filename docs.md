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

## HDR composition

The script `hdr.py` combines the stacks produced by `integration.py`, located in `stacks_dir`, into a single high-dynamic-range image `hdr.fits` written to `hdr_dir`. It expects one stack per group, so the same `group_keywords` must be used as during integration.

Each stack only sees part of the corona : the longest exposure saturates over the inner corona, while the shortest one drowns the outer corona in noise. Walking from the longest exposure to the shortest, every stack is fitted onto the brightness scale of the previous one and added to a weighted average, so that each pixel is drawn from the exposures that actually recorded it within their usable range.

The chain has to run from the longest exposure to the shortest, but the stacks are sorted by the raw value of the grouping keywords, which only tracks brightness when the keyword is an exposure setting. Rather than assume it, each stack is compared against the previous one as it is read, and the script stops if it turns out not to be the fainter of the two.

### Clipping thresholds

A sensor only records brightness faithfully over part of its range : it clips at the top and drowns in noise at the bottom. Neither end fails abruptly, since values above ~80-90% of the clipping level are already compressed and values near 0 are already noise-dominated, so both have to be rejected with some margin for the composite to come out smooth and realistic.

Where that clipping level sits depends on the imaging system, and calibration (bias subtraction, flat division) shifts it further, by a different amount for every pixel. It is therefore measured rather than configured : the longest exposure saturates over the inner corona, so the brightest value it holds is the level at which the system clips. The four parameters below are <b>fractions of that measured level</b>, not absolute pixel values, so the same configuration transfers between cameras and between calibration setups.

A pixel counts as usable only if <b>every</b> colour channel is in range : the dimmest channel above the low threshold, and the brightest one below the high threshold. The weighting function is defined by 4 parameters :
- `high_threshold`, `high_smoothness` : fractions of the measured saturation level. The weight function is equal to 0 for pixel values above `high_threshold`, and equal to 1 below `high_threshold`-`high_smoothness`. Between the two, it is a simple linear interpolation.
- `low_threshold`, `low_smoothness` : analogous to `high_threshold` and `high_smoothness`.

The longest and the shortest exposure are treated as special cases : the longest one keeps its dark pixels and the shortest one keeps its bright pixels, since in each case no other stack measures that part of the corona.

Consecutive exposures must overlap in usable range, otherwise there is nothing to fit them against. How far apart they may be is set by the range the thresholds cover, log2(`high_threshold`/`low_threshold`) : the default 0.025 and 0.8 cover <b>5 stops</b>, so exposures more than 5 stops apart share no usable pixel at all, and <b>3 stops or less is strongly recommended</b>. Widen the range when consecutive exposures do not overlap enough, and tighten it when non-linearity shows through in the composite, as a ring at the handover radius between two exposures.

All of the above assumes the images carry no offset : a pixel that recorded nothing reads 0, which is what Umbra's own calibration produces once the bias is subtracted. A leftover pedestal lifts every pixel by the same amount while the thresholds keep counting from zero, so both have to be <b>raised</b>. It matters most for `low_threshold` : left alone, a pixel holding nothing but the pedestal clears the low cut and is weighted as though it had recorded something.

### Brightness equalization

Before being combined, each stack is fitted onto the brightness scale of the previous one, through an affine map whose offset and slope both vary with the angle around the moon. The fit absorbs the exposure ratio itself, along with the transparency and sky gradient differences that a nominal exposure ratio cannot describe. It is computed on the pixels that are usable in both stacks, excluding the moon.

### Weighting and output

Each stack contributes in proportion to its exposure time, which is the inverse-variance weighting of a photon-noise-limited signal once every stack has been brought onto a common brightness scale. Exposure time alone, and not the exposure-gain product : raising the gain amplifies signal and noise together and buys no photons. When a stack records no exposure time (neither `EXPTIME` nor `EXPOSURE`), all stacks are weighted equally instead and a warning is printed.

Fitting onto the longest exposure's scale pushes the inner corona well past 1, so the composite is rescaled to [0,1] before being written. The divisor is recorded in the `HDRSCALE` header keyword : multiply by it to recover the values on the longest exposure's scale.

- `save_weights` : when true, the weight map of each group is also written to `hdr_dir`. Useful to tune the four clipping parameters, since it shows exactly which pixels each exposure contributed.

