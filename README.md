# DataSAXS Template

Prototype interface for SAXS data processing from ID13, ID02, and XENOCS beamlines.

## Launching the application

```bash
cd DataSAXS_Template
python3 main.py

On macOS with Python 3.14:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 main.py
```

## Installing dependencies

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pip install -r requirements.txt
```

## Structure

```text
DataSAXS_Template/
├── main.py
├── requirements.txt
├── README.md
├── tabs/
│   ├── __init__.py
│   ├── find_centre_tab.py
│   ├── xenocs_cave_tab.py
│   └── id13_cave_tab.py
├── utils/
│   └── __init__.py
└── assets/
```
