# RbxToMii source code

The source code of RbxToMii (Python). `RbxToMii.pyw` is the window, `rbxtomii.py` does the work: it loads a Roblox avatar
(Roblox's public web APIs, no login; login-only parts from Roblox's cache on the PC), builds the game files
(`build_roblox_models.py`, `rbxchar.py`, `miimap.py`, ...), writes them into the Ryujinx mod folder `RbxToMii` and
dresses the chosen Mii in the save (`ugcsave.py`).

## Run or build it

1. Windows 10/11, Python 3.13 (64-bit). `pip install -r requirements.txt`
2. The folder `templates` from the release zip (numbers and template files from the game) belongs next to the
   program. To run from source, put it here as `../rbxtomii/templates` (see `TOOL_DIR` in `rbxtomii.py`), or build:
3. `python build_exe.py <the release's templates folder> <out folder>` makes `RbxToMii.exe` with everything it needs.

## License

Same as RbxToMii: free to use and to share; if you make your own or a better tool based on this one, you must credit
me - see [LICENSE.txt](../LICENSE.txt).

This mod was created only thanks to Claude Opus 5.5
