Umbra: Total Solar Eclipse Image Processing Software
==============

<p align="center">
  <img src="docs/assets/moon_registration.gif" alt="Moon registration" width="45%"/>
  <img src="docs/assets/sun_registration.gif" alt="Sun registration" width="45%"/>
  <br/>
  <sub><em>Images provided by @astro_uri on Instagram</em></sub>
</p>

# Install
To use Umbra, download the appropriate `.zip` file for your operating system from the [latest release](https://github.com/LudoRey/umbra/releases/latest). Inside the extracted folder, you will find the executable. **Do not move the executable out of the folder**. If you want to have a standalone file on your desktop, create a shortcut.

Note: some antivirus software may flag the executable as a false positive; if this happens, add it to your antivirus exclusion list.

# Quick start

The software should be self-explanatory : you can hover over almost any element to see a tooltip explaining its function. To make things even easier, this quick guide will help you get started.

## Registration

Before using Umbra for image registration, you must convert your (debayered) images to FITS format, and place them in a single folder. **Do not organize your images into subfolders**.


1. Open the folder containing your FITS images.
<img src="docs/assets/quick_start/open_folder.gif" width="100%">

2. Select a reference <img src="docs/assets/quick_start/target.png" height="16px"> and an anchor <img src="docs/assets/quick_start/anchor.png" height="16px">. Hover over the fileviewer header icons to see tips on how to make your selection.
<img src="docs/assets/quick_start/select_reference_and_anchor.gif" width="100%">

3. Set your image scale and click on start.
<img src="docs/assets/quick_start/set_image_scale_and_run.gif" width="100%">

The registered images are saved in the folders specified by the options "Moon-registered folder" and "Sun-registered folder". By default, they are located in the parent folder of the input folder.

# Join the Community

Have questions, want to discuss solar eclipses, or get help with Umbra? 
[Join our Discord server](https://discord.gg/Ayu7qaZETq).

# Source code

The source code for the algorithms is available if you want to run them without using the GUI. See the [documentation](docs.md) for more information. 