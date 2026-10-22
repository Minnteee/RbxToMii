# RbxToMii

**Put a Roblox avatar on a Mii in _Tomodachi Life: Living the Dream_.**
A Windows tool for the **Ryujinx** emulator.

> **Version 1.1 – final version, for game version 1.0.4.** Tested with **Ryujinx Canary 1.3.269** (firmware 22.1.0).
> The **source code** is in the folder [`src`](src).
> This repository does not take contributions: please no pull requests, issues or bug reports.

## What it does

Type a Roblox username, pick a Mii from your island, press one button, and the Mii becomes that Roblox character in the game. It gets the real Roblox body, clothes and accessories, and the real Roblox head (Dynamic Head with its face) whenever possible, in real Roblox proportions. The character moves with the game's own animations: walking, sitting, eating with its hands and so on.

How it works: the game can only draw Mii bodies. So RbxToMii turns a full-body costume and a full head mask of the game into the Roblox character and puts both on your Mii. It saves them as a normal Ryujinx mod (folder `RbxToMii`) and changes your save so that the Mii wears them.

Made for **Ryujinx** and tested with **Ryujinx Canary 1.3.269** (firmware 22.1.0) on Windows. **I can't guarantee that it works with other emulators.** RbxToMii 1.1 is made for game version **1.0.4** (the current one). Still on game version 1.0.1? Use RbxToMii 1.0.0 from the releases.

## Download

From [**Releases**](https://github.com/Minnteee/RbxToMii/releases/latest): `RbxToMii-1.1.0.zip`

## What's new

- **1.1.0** (final version) – Updated for game version **1.0.4**. The game's size table changes with every update; with the old one from 1.0.1, game version 1.0.4 crashed right after starting (reported on GameBanana – thanks!). RbxToMii now also warns when your save is from another game version, and it respects mods you turned off in Ryujinx (Manage Mods). Tested with Ryujinx Canary 1.3.269.
  - **Faces:** Roblox heads show their real face again. Many Roblox heads have exactly the same size, and RbxToMii sometimes took the wrong one from Roblox's cache (the face then had no mouth); now the head's texture decides. Heads with a SurfaceAppearance show their face, older heads show the avatar's mood (smile, smirk, …) too, and when the mood can't be read it shows Roblox's default smile. A stand-in head (when the real one is login-only) now gets Roblox's classic face with eyes instead of only a smile.
  - **Arms:** arms move like Roblox's own block arms. Before, arm blocks got crumpled in strong poses (clapping, eating, hands on hips).
  - Very old classic T-shirts, shirts and pants (from Roblox's first years, like builderman's) are found again.
- **1.0.0** – First version (game version 1.0.1, Ryujinx 1.3.287).

## What you need

| | What | Notes |
|---|---|---|
| ✔ | Windows 10 or 11, 64-bit | |
| ✔ | Ryujinx (tested: Ryujinx Canary 1.3.269, firmware 22.1.0) | set up with your own game dump, keys and firmware |
| ✔ | Your own copy of Tomodachi Life: Living the Dream **with update 1.0.4** | in Ryujinx: right-click the game → *Manage Title Updates* → select 1.0.4. Other versions don't work with this release. |
| ✔ | An island with at least one Mii, saved once | RbxToMii changes this save |
| ✔ | Internet connection | RbxToMii loads avatars from Roblox's public servers (no login, no password) |
| ➕ optional | Roblox (the app) or **Roblox Studio**, on the same PC | only for login-only items, see [Missing items](#missing-items-roblox-studio) |

You do **not** need Python, Visual C++ downloads, other mods or any other installs: RbxToMii brings everything it needs.

## Install

1. Download `RbxToMii-1.1.0.zip` and unzip it (right-click → *Extract All*) to a folder you can write to, for example your Desktop or Documents (not *Program Files*). Don't start it from inside the zip.
2. Start **`RbxToMii.exe`**. Windows may say *"Windows protected your PC"* the first time, because the program is not signed. Click **More info → Run anyway**. Some antivirus programs also distrust new unsigned programs; if yours blocks RbxToMii, allow it there.

That's all. RbxToMii makes its Ryujinx mod folder itself the first time you add a character.

## How to use

1. In the game, make a Mii for the Roblox character (any look, any name) or pick one you already have. Let the game save, then **close Ryujinx**.
2. Start **`RbxToMii.exe`**. At the bottom it shows the Ryujinx folder it found (green ✓). If it doesn't find it, press **Change …** and choose the Ryujinx data folder (`%APPDATA%\Ryujinx`, or `portable` for the portable version).
3. Type the **Roblox username** and press **Load**. Left: the avatar on Roblox. Right: how it will look in the game.
4. If the tool shows **"Missing: …"**, some items are login-only. See [Missing items](#missing-items-roblox-studio).
5. Under **Mii in your island**, choose the Mii that should become the character.
6. Press **Add to Tomodachi Life**. Ryujinx must be closed for this, because RbxToMii writes into your save.
7. Start the game: the Mii now looks like the Roblox character.

### Good to know

- RbxToMii's mod folder is `%APPDATA%\Ryujinx\mods\contents\010051f0207b2000\RbxToMii` (in Ryujinx: right-click the game → *Open Mods Directory*). It has to stay enabled under *Manage Mods*.
- There are **7 spots**. Each spot turns one full-body suit and one full head mask of the game into the Roblox suit and head, **for every Mii that wears them and in the shop**. In the wardrobe they keep the game's names:

  | Spot | Suit | Head |
  |---|---|---|
  | 1 | Astronaut suit | Superhero helmet |
  | 2 | Sci-fi suit | Wrestling mask |
  | 3 | Retro diving suit | Skull mask |
  | 4 | Lion suit | Knight helmet |
  | 5 | Tiger suit | Paper bag |
  | 6 | Horse suit | Gladiator helmet |
  | 7 | Monkey suit | Pumpkin head |

  So a Mii that already wears one of these items looks like the Roblox character too. When all 7 spots are used, RbxToMii asks which character to replace. Adding the same character again updates it.
- The Roblox suit and head count as bought, so you can put them on other Miis in the wardrobe too.
- Before every change RbxToMii makes a backup in `RbxToMii\backups`: the replaced files and your save.
- You see the changes after (re)starting the game.
- Other mods that change the game's clothing table (`RSDB/ClothModelParam…`) or size table (`System/Resource/ResourceSizeTable…`) can't be used together with RbxToMii.
- **The game crashes right after starting?** Then your game version doesn't fit this RbxToMii (1.1 is for game version 1.0.4). Check *Manage Title Updates* in Ryujinx, or turn the RbxToMii mod off under *Manage Mods*. After a new game update, RbxToMii needs an update too.

## Missing items (Roblox Studio)

Roblox only gives some items to logged-in users: **Dynamic Heads** (the real 3D head with its face), a lot of **layered clothing**, and some hair and accessories. RbxToMii **never logs in** and never asks for a password. Instead it takes these items from Roblox's cache on **your PC** (`%LOCALAPPDATA%\Roblox`). Everything that Roblox or Roblox Studio has shown once on this PC is stored there. So if something is missing, let Roblox show the character once.

**Way 1 – Roblox app:** see the character once in Roblox on this PC (for example, play a game together). Then press **Load** again.

**Way 2 – Roblox Studio (works for any player):**

1. Install **Roblox Studio** (free, [create.roblox.com](https://create.roblox.com)) and log in with your Roblox account.
2. Create a new place: **Baseplate**.
3. Open the **Command Bar**: *Script* tab → *Command Bar*, or press **Ctrl+9**.
4. Copy this line into the Command Bar, replace `USERNAME` with the Roblox username (keep the quotes) and press **Enter** (in the newer multi-line Command Bar: **Ctrl+Enter** or the *Run* button):
   ```lua
   local P = game:GetService("Players") local name = "USERNAME" local m = P:CreateHumanoidModelFromDescriptionAsync(P:GetHumanoidDescriptionFromUserIdAsync(P:GetUserIdFromNameAsync(name)), Enum.HumanoidRigType.R15) m.Name = name m.Parent = workspace
   ```
   The character appears on the baseplate. To show it, Studio downloads all of its items into the cache on your PC, the login-only ones too.
5. In the **Explorer**, find the new model with the player's name under *Workspace*. Right-click it → **Save to File…** (in some Studio versions: *Save / Export → Save to File…*) and save it as a **`.rbxm`** file, for example on your Desktop.
   Saving the whole place works too: *File → Save to File As…* → `.rbxl`. RbxToMii then asks which character to use.
6. In RbxToMii press **Open Roblox Studio file …** and choose that file. RbxToMii reads the exact character from the file and takes the login-only items from the cache that Studio just filled.
7. Go on with step 5 of [How to use](#how-to-use).

Do this on the **same PC** where you use RbxToMii, because the cache only exists on that PC. Your Roblox login stays in Roblox Studio; RbxToMii never sees it.

## Uninstall

Close Ryujinx and delete the `RbxToMii` mod folder (or disable it under *Manage Mods*). Miis that wore Roblox items then wear the game's original items again. The program folder can simply be deleted.

## Limits

- The game can only draw Mii bodies. So a Roblox character is worn as a full-body costume plus a full head mask on the Mii skeleton, scaled as a whole. The arm blocks turn around the Mii's joints, which sit a bit further in than Roblox's: in some poses arm blocks overlap a little at the shoulder or elbow (Roblox characters do that too).
- In the resident list and the Mii bar, Roblox heads look small, because the game frames these pictures for a Mii head.
- If the game later dresses a Roblox Mii differently, put the Roblox suit and head on it again in the wardrobe.

## Source code

The source code (Python) is in the folder [`src`](src); how to build `RbxToMii.exe` from it is in [src/README.md](src/README.md). This is the final version: the repository doesn't take pull requests or issues.

## License and credits

Free to use and to share. **If you make your own or a better tool based on this one, you must credit me.** The exact terms are in [LICENSE.txt](LICENSE.txt).

Fan project, not affiliated with Nintendo or Roblox. You need your own copy of the game; RbxToMii does not include the game, keys or firmware, only the few game files it uses as templates (costume and mask models, the clothing table and the size table). Tomodachi Life: Living the Dream belongs to Nintendo. Roblox avatars and items belong to Roblox Corporation and their creators.

This mod was created only thanks to Claude Opus 5.5
