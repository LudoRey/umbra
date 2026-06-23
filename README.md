Umbra: Total Solar Eclipse Image Processing Software
==============

<p align="center">
  <img src="docs/assets/moon_registration.gif" alt="Moon registration" width="45%"/>
  <img src="docs/assets/sun_registration.gif" alt="Sun registration" width="45%"/>
  <br/>
  <sub><em>Images provided by @astro_uri on Instagram</em></sub>
</p>

# Install

## Windows

Download `umbra-<version>-windows-x64-setup.exe` from the [latest release](https://github.com/LudoRey/umbra/releases/latest) and open it. Because the app is not yet signed, Windows SmartScreen may warn that it "protected your PC"; click **More info**, then **Run anyway**.

**Keep the default install location**. The installer lets you choose where to install Umbra, but keep the default (`%LOCALAPPDATA%\Umbra`). Automatic updates work by replacing Umbra's own folder, which is only possible where it can write without administrator rights — installing elsewhere (in particular `Program Files`) will break updates.

## macOS (Apple Silicon)

Download `umbra-<version>-macos-arm64.pkg` from the [latest release](https://github.com/LudoRey/umbra/releases/latest) and open it. Because the app is not yet notarized, macOS blocks it on first open with a message that it "could not verify" the package. To allow it:

1. Click **Done** on the warning (not *Move to Trash*).
2. Open **System Settings → Privacy & Security**.
3. Scroll down to the **Security** section, find the message about `umbra-<version>-macos-arm64.pkg`, and click **Open Anyway**.
4. Confirm **Open Anyway** and enter your password when prompted.

Open Umbra from **Spotlight** (press ⌘ + Space and type "Umbra") or from **Launchpad**. The installer puts it in the Applications folder inside your Home folder, which is not the Applications shortcut in the Finder sidebar (that one is the system-wide folder). Leave it there: moving it elsewhere, including into the system Applications folder, will break automatic updates.

# Join the Community

Have questions, want to discuss solar eclipses, or get help with Umbra? 
[Join our Discord server](https://discord.gg/Ayu7qaZETq).

# Source code

The source code for the algorithms is available if you want to run them without using the GUI. See the [documentation](docs.md) for more information. 