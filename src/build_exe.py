"""Build RbxToMii.exe (PyInstaller, one folder) from this source folder.
usage: python build_exe.py <templates folder> <out folder>
<templates folder>: the folder "templates" of the RbxToMii release zip (numbers and template files from the game)."""
import os, sys, shutil, subprocess

SRC = os.path.dirname(os.path.abspath(__file__))
TEMPLATES, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
BUILD = os.path.join(os.path.dirname(OUT), '_rbxtomii_build')     # outside the source folder
MODULES = ['rbxtomii', 'rbxapi', 'rbxfile', 'rbxassemble', 'rbxcache', 'rbxmesh', 'rbxchar', 'rbxmood', 'rbx_substitutes', 'build_roblox_models', 'bfres', 'bfres_write', 'bntx', 'bntx_write', 'byml_write', 'tlfmt', 'miimap', 'miipose', 'fska', 'restbl', 'chinfit', 'rrender', 'preview_pose', 'ugcsave', 'tlsave', 'hashnames', 'parser', 'ugctex', 'preview_game_model']
cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--windowed', '--onedir', '--name', 'RbxToMii',
       '--distpath', os.path.join(BUILD, 'dist'), '--workpath', os.path.join(BUILD, 'work'), '--specpath', BUILD,
       '--paths', SRC]
for m in MODULES:
    cmd += ['--hidden-import', m]
cmd += ['--hidden-import', 'DracoPy', '--hidden-import', 'texture2ddecoder', '--hidden-import', 'zstandard',
        os.path.join(SRC, 'RbxToMii.pyw')]
subprocess.run(cmd, check=True, cwd=SRC)
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
shutil.copytree(os.path.join(BUILD, 'dist', 'RbxToMii'), OUT)
shutil.copytree(TEMPLATES, os.path.join(OUT, 'templates'))
print('done:', OUT)
