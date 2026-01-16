# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for generate_wing.py

This spec file is used to build a standalone executable for the
wing geometry generation script using --onedir mode for faster startup.
"""

block_cipher = None

a = Analysis(
    ['../generate_wing.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib',
        'numpy.lib.format',
        'scipy',
        'scipy.interpolate',
        'scipy.spatial',
        'trimesh',
        'trimesh.resources',
        'trimesh.exchange',
        'trimesh.exchange.stl',
        'manifold3d',
        'networkx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='generate_wing',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='generate_wing',
)
