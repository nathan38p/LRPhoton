@echo off
title LRPhoton Installer
setlocal

echo ==========================================
echo          INSTALLATION LRPhoton
echo ==========================================
echo.

set "PYTHON_EXE=python"

echo Installation / mise a jour de Python 3.14.5 x64...
winget install --id Python.Python.3.14 -v 3.14.5 --architecture x64 -e

if errorlevel 1 (
    echo.
    echo ERREUR : Python 3.14.5 x64 n'a pas pu etre installe avec winget.
    echo Installez Python 3.14.5 x64 manuellement puis relancez Install.bat.
    echo.
    pause
    exit /b 1
)

echo.
echo Verification de Python...

%PYTHON_EXE% --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo ERREUR : Python n'est pas detecte apres installation.
    echo Fermez puis rouvrez l'invite de commande, ou redemarrez Windows, puis relancez Install.bat.
    echo.
    pause
    exit /b 1
)

echo Python detecte.
echo.

echo Version Python :
%PYTHON_EXE% --version

echo.
echo Architecture Python :
%PYTHON_EXE% -c "import platform; print(platform.machine())"

echo.

set "SOURCE=%~dp0"
set "DEST=C:\Program Files\LRPhoton"

echo Creation du dossier...
mkdir "%DEST%" >nul 2>&1

echo.
echo Copie des fichiers...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Copy-Item -Path '%SOURCE%*' -Destination '%DEST%' -Recurse -Force"

echo.
echo Installation des dependances...

%PYTHON_EXE% -m pip install --upgrade pip

%PYTHON_EXE% -m pip install ^
PySide6 ^
numpy ^
matplotlib ^
h5py ^
requests ^
hdf5plugin ^
fabio

echo.
echo Creation du lanceur bureau...

(
echo @echo off
echo cd /d "C:\Program Files\LRPhoton"
echo python main.py
) > "%USERPROFILE%\Desktop\LRPhoton.bat"

echo.
echo Creation icone...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = '%USERPROFILE%\Desktop\LRPhoton.bat'; ^
$Shortcut.WorkingDirectory = 'C:\Program Files\LRPhoton'; ^
$Shortcut.IconLocation = 'C:\Program Files\LRPhoton\assets\LRPhoton.ico'; ^
$Shortcut.Save()"

echo.
echo ==========================================
echo Installation terminee
echo ==========================================
pause