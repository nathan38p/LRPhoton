@echo off
title LRPhoton Installer

echo ==========================================
echo          INSTALLATION LRPhoton
echo ==========================================
echo.

:: =========================================================
:: INSTALLATION PYTHON 3.14 x64 SI ABSENT
:: =========================================================

echo Verification de Python 3.14 x64...

py -3.14-64 --version >nul 2>&1

if errorlevel 1 (

    echo.
    echo Python 3.14 x64 non detecte.
    echo Telechargement de Python 3.14 x64 depuis python.org...
    echo.

    set "PYTHON_INSTALLER=%TEMP%\python-3.14.5-amd64.exe"

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.14.5/python-3.14.5-amd64.exe' -OutFile '%PYTHON_INSTALLER%'"

    echo.
    echo Installation de Python 3.14 x64...
    echo.

    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1

    echo.
    echo Verification installation...
    echo.

    py -3.14-64 --version >nul 2>&1

    if errorlevel 1 (
        echo.
        echo ERREUR :
        echo Python 3.14 x64 n'a pas ete detecte.
        echo.
        echo Redemarrez Windows puis relancez Install.bat
        echo.
        pause
        exit
    )
)

echo Python 3.14 x64 detecte.
echo.

:: =========================================================
:: DOSSIER INSTALLATION
:: =========================================================

set "SOURCE=%~dp0"
set "DEST=C:\Program Files\LRPhoton"

echo Creation du dossier...
mkdir "%DEST%" >nul 2>&1

echo.
echo Copie des fichiers...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"Copy-Item -Path '%SOURCE%*' -Destination '%DEST%' -Recurse -Force"

echo.
echo Installation des dependances...

py -3.14-64 -m pip install --upgrade pip

py -3.14-64 -m pip install ^
PySide6 ^
numpy ^
matplotlib ^
h5py ^
fabio ^
requests ^
hdf5plugin

echo.
echo Creation du raccourci bureau...

(
echo @echo off
echo cd /d "C:\Program Files\LRPhoton"
echo py -3.14-64 main.py
) > "%USERPROFILE%\Desktop\LRPhoton.bat"

echo.
echo Creation icone...

powershell -Command ^
"$WshShell = New-Object -comObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\LRPhoton.lnk'); ^
$Shortcut.TargetPath = '%USERPROFILE%\Desktop\LRPhoton.bat'; ^
$Shortcut.IconLocation = 'C:\Program Files\LRPhoton\assets\LRPhoton.ico'; ^
$Shortcut.Save()"

echo.
echo ==========================================
echo Installation terminee
echo ==========================================
echo.
echo Logiciel :
echo C:\Program Files\LRPhoton
echo.
echo Raccourci cree sur le bureau :
echo LRPhoton
echo.

pause