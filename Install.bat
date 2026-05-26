@echo off
title LRPhoton Installer
setlocal

echo ==========================================
echo          INSTALLATION LRPhoton
echo ==========================================
echo.

set "PYTHON_EXE=python"

echo Installation / update of Python 3.14.5 x64...
winget install --id Python.Python.3.14 -v 3.14.5 --architecture x64 -e

if errorlevel 1 (
    echo.
    echo ERROR: Python 3.14.5 x64 could not be installed with winget.
    echo Install Python 3.14.5 x64 manually, then relaunch Install.bat.
    echo.
    echo Press any key to close Install.bat...
    pause
    exit /b 1
)

echo.
echo Checking Python...

%PYTHON_EXE% --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo Refreshing environment variables...

    set "PATH=%PATH%;%LocalAppData%\Microsoft\WindowsApps"

    %PYTHON_EXE% --version >nul 2>&1

    if errorlevel 1 (
        echo.
        echo ERROR: Python is still not detected after installation.
        echo Restart Windows then relaunch Install.bat.
        echo.
        echo Press any key to close and relaunch Install.bat...
        pause
        exit /b 1
    )
)

echo Python detected.
echo.

echo Python version:
%PYTHON_EXE% --version

echo.
echo Python architecture:
%PYTHON_EXE% -c "import platform; print(platform.machine())"

echo.

set "SOURCE=%~dp0"

net session >nul 2>&1
if errorlevel 1 (
    echo No administrator rights detected.
    echo LRPhoton will be installed for the current user.
    set "DEST=%LOCALAPPDATA%\Programs\LRPhoton"
) else (
    echo Administrator rights detected.
    echo LRPhoton will be installed for all users.
    set "DEST=C:\Program Files\LRPhoton"
)

for %%I in ("%SOURCE%.") do set "SOURCE_FULL=%%~fI"
for %%I in ("%DEST%") do set "DEST_FULL=%%~fI"

echo Creating installation folder...
mkdir "%DEST%" >nul 2>&1

echo.
if /I "%SOURCE_FULL%"=="%DEST_FULL%" (
    echo Install.bat is already running from the installation folder.
    echo Skipping file copy.
) else (
    echo Copying files...
    robocopy "%SOURCE%" "%DEST%" /E /XD .git __pycache__ .venv venv build dist /XF .DS_Store /NFL /NDL /NJH /NJS /NP

    if errorlevel 8 (
        echo.
        echo ERROR: File copy failed.
        echo Make sure Install.bat is running as administrator.
        echo.
        echo If you do not have administrator rights, install LRPhoton manually:
        echo 1. Go to the LRPhoton GitHub page.
        echo 2. Click the green Code button, then Download ZIP.
        echo 3. Extract the ZIP.
        echo 4. Copy the contents of the LRPhoton folder into:
        echo    C:\Program Files\LRPhoton
        echo    or C:\Programmes\LRPhoton if this is your installation folder.
        echo.
        echo Press any key to close Install.bat...
        pause
        exit /b 1
    )
)

echo.
echo Installing dependencies...

%PYTHON_EXE% -m pip install --upgrade pip

%PYTHON_EXE% -m pip install ^
PySide6 ^
numpy ^
matplotlib ^
h5py ^
requests ^
hdf5plugin ^
fabio ^
scipy ^
pyFAI

if errorlevel 1 (
    echo.
    echo ERROR: Some dependencies could not be installed.
    echo.
    echo Press any key to close Install.bat...
    pause
    exit /b 1
)

echo.
echo Creating desktop shortcut...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = 'pythonw.exe'; ^
$Shortcut.Arguments = 'main.py'; ^
$Shortcut.WorkingDirectory = 'C:\Program Files\LRPhoton'; ^
$Shortcut.IconLocation = 'C:\Program Files\LRPhoton\assets\LRPhoton.ico'; ^
$Shortcut.Save()"

if errorlevel 1 (
    echo.
    echo ERROR: Could not create desktop shortcut.
    echo.
    echo Press any key to close Install.bat...
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Installation complete
echo ==========================================
echo.
echo Software:
echo C:\Program Files\LRPhoton
echo.
echo Desktop shortcut created:
echo LRPhoton
echo.
echo Press any key to close Install.bat...
pause
