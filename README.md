# LRPhoton

## Overview

LRPhoton is a Python application for SAXS/WAXS data processing and visualization dedicated to laboratory and synchrotron scattering data, including XENOCS laboratory SAXS systems and ESRF beamlines ID13 and ID02.

## Windows installation

1. Click the green Code button, then select Download ZIP.
2. Extract the ZIP archive.
3. Open the extracted folder.
4. Double click on Install.bat.

The installer will:

* install Python automatically if it is missing
* install all required Python dependencies
* copy the application into `%LOCALAPPDATA%\Programs\LRPhoton`
* create a desktop shortcut automatically

Important:

* Administrator rights are not required.
* If Python has just been installed, relaunch Install.bat once after the Python installation finishes.
* If Windows asks how to open the file, choose Command Prompt or Windows Terminal.

The installed application folder is usually:

```text
C:\Users\<your-user-name>\AppData\Local\Programs\LRPhoton
```

## MacOS Installation

1. Install Python 3.14.5 from: https://www.python.org/downloads/
2. Click the green Code button, then select Download ZIP.
3. Move the extracted LRPhoton folder into /Applications.
4. Double click `Create_macOS_app.command`.
5. Launch LRPhoton using the `LRPhoton.app` application created in /Applications.

The macOS launcher is only a small application that opens:

```text
/Applications/LRPhoton/main.py
```

The real LRPhoton files stay in `/Applications/LRPhoton`, so the automatic update system can still replace the Python files in that folder.

## Update System

At startup, LRPhoton automatically checks for updates from the GitHub repository and downloads modified internal files when a newer version is available.
