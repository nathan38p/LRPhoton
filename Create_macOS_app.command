#!/bin/bash
set -euo pipefail

APP_DIR="/Applications/LRPhoton"
LAUNCHER_APP="/Applications/LRPhoton.app"
LOG_FILE="$HOME/Library/Logs/LRPhoton.log"

if [ ! -f "$APP_DIR/main.py" ]; then
    echo "ERROR: $APP_DIR/main.py was not found."
    echo "Move the LRPhoton folder into /Applications first:"
    echo "$APP_DIR"
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

rm -rf "$LAUNCHER_APP"

osacompile -o "$LAUNCHER_APP" -e "do shell script \"cd '$APP_DIR' && /usr/bin/env python3 main.py >> '$LOG_FILE' 2>&1 &\""

if [ -f "$APP_DIR/assets/LRPhoton.icns" ]; then
    cp "$APP_DIR/assets/LRPhoton.icns" "$LAUNCHER_APP/Contents/Resources/applet.icns"
    /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile applet" "$LAUNCHER_APP/Contents/Info.plist" >/dev/null 2>&1 || true
    touch "$LAUNCHER_APP"
fi

echo "LRPhoton launcher created:"
echo "$LAUNCHER_APP"
echo
echo "You can now launch LRPhoton from /Applications."
echo
read -r -p "Press Enter to close..."
