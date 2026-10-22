"""RbxToMii - put a Roblox character on a Mii in Tomodachi Life: Living the Dream (Ryujinx)."""
import os, sys, threading, queue, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import rbxtomii as R

TITLE = 'RbxToMii'
PIC = 300
NO_MII = '(choose a Mii)'


class App:
    def __init__(self, root):
        self.root = root
        root.title(TITLE)
        root.geometry('700x700')
        root.minsize(640, 680)
        self.q = queue.Queue()
        self.busy = False
        self.env = None
        self.ch = None
        self.built = {}
        self.miis = []
        self._imgs = {}
        self._ui()
        self.root.after(100, self._poll)
        self.root.after(150, self.detect)
        if os.environ.get('RBXTOMII_TEST_NAME'):      # automatic run (tests)
            self.name_var.set(os.environ['RBXTOMII_TEST_NAME'])
            self.root.after(1200, self.load_player)

    # ------------------------------------------------------------ layout
    def _ui(self):
        top = ttk.Frame(self.root, padding=(16, 14, 16, 6))
        top.pack(fill='x')
        ttk.Label(top, text='Put a Roblox character on a Mii in Tomodachi Life', font=('Segoe UI', 13, 'bold')).pack(anchor='w')

        row = ttk.Frame(self.root, padding=(16, 6))
        row.pack(fill='x')
        ttk.Label(row, text='Roblox username:', font=('Segoe UI', 10)).pack(side='left')
        self.name_var = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.name_var, width=26, font=('Segoe UI', 11))
        e.pack(side='left', padx=8)
        e.bind('<Return>', lambda ev: self.load_player())
        e.focus_set()
        self.bt_load = ttk.Button(row, text='Load', command=self.load_player)
        self.bt_load.pack(side='left')
        self.bt_file = ttk.Button(row, text='Open Roblox Studio file …', command=self.load_file)
        self.bt_file.pack(side='right')

        pics = ttk.Frame(self.root, padding=(16, 8))
        pics.pack(fill='x')
        self.pic_rbx = self._pic(pics, 'In Roblox', 0)
        self.pic_game = self._pic(pics, 'In the game', 1)
        pics.columnconfigure(0, weight=1)
        pics.columnconfigure(1, weight=1)

        self.info = ttk.Label(self.root, text='Type a Roblox username and press Load.', padding=(16, 2),
                              font=('Segoe UI', 9), wraplength=660, justify='left')
        self.info.pack(fill='x')

        mrow = ttk.Frame(self.root, padding=(16, 8, 16, 0))
        mrow.pack(fill='x')
        ttk.Label(mrow, text='Mii in your island:', font=('Segoe UI', 10)).pack(side='left')
        self.mii_var = tk.StringVar(value=NO_MII)
        self.mii_box = ttk.Combobox(mrow, textvariable=self.mii_var, state='readonly', width=36, font=('Segoe UI', 10),
                                    postcommand=self.refresh_miis)
        self.mii_box.pack(side='left', padx=8)
        self.mii_box.bind('<<ComboboxSelected>>', lambda ev: self._update_add())

        act = ttk.Frame(self.root, padding=(16, 10))
        act.pack(fill='x')
        style = ttk.Style()
        style.configure('Big.TButton', font=('Segoe UI', 11, 'bold'), padding=8)
        self.bt_add = ttk.Button(act, text='Add to Tomodachi Life', style='Big.TButton', command=self.add, state='disabled')
        self.bt_add.pack(fill='x')

        self.prog = ttk.Progressbar(self.root, mode='determinate', value=0)
        self.prog.pack(fill='x', padx=16)
        self.status = ttk.Label(self.root, text='', padding=(16, 4), foreground='#444', font=('Segoe UI', 9))
        self.status.pack(fill='x')

        bottom = ttk.Frame(self.root, padding=(16, 2, 16, 10))
        bottom.pack(fill='x', side='bottom')
        self.env_lbl = ttk.Label(bottom, text='', font=('Segoe UI', 8), wraplength=560, justify='left')
        self.env_lbl.pack(side='left', fill='x', expand=True)
        ttk.Button(bottom, text='Change …', command=self.pick_ryujinx, width=10).pack(side='right')

    def _pic(self, parent, title, col):
        fr = ttk.Frame(parent)
        fr.grid(row=0, column=col, padx=6, sticky='n')
        ttk.Label(fr, text=title, font=('Segoe UI', 9, 'bold')).pack()
        lb = tk.Label(fr, bg='#eef1f5', width=PIC, height=PIC, bd=0)
        lb.pack()
        blank = ImageTk.PhotoImage(Image.new('RGB', (PIC, PIC), (238, 241, 245)))
        self._imgs['blank%d' % col] = blank
        lb.configure(image=blank)
        return lb

    def show(self, label, img, key):
        im = img.copy()
        im.thumbnail((PIC, PIC))
        bg = Image.new('RGB', (PIC, PIC), (238, 241, 245))
        bg.paste(im, ((PIC - im.size[0]) // 2, (PIC - im.size[1]) // 2))
        tkim = ImageTk.PhotoImage(bg)
        self._imgs[key] = tkim
        label.configure(image=tkim)

    # ------------------------------------------------------------ background work
    def say(self, msg):
        self.q.put(('status', msg))

    def work(self, fn, done=None):
        if self.busy:
            return
        self.busy = True
        self.prog.configure(mode='indeterminate')
        self.prog.start(12)
        for b in (self.bt_load, self.bt_file, self.bt_add):
            b.configure(state='disabled')

        def run():
            try:
                self.q.put(('done', (done, fn(), None)))
            except Exception as e:
                traceback.print_exc()
                self.q.put(('done', (done, None, e)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == 'status':
                    self.status.configure(text=val)
                elif kind == 'done':
                    done, res, err = val
                    self.busy = False
                    self.prog.stop()
                    self.prog.configure(mode='determinate', value=0)
                    self.bt_load.configure(state='normal')
                    self.bt_file.configure(state='normal')
                    self._update_add()
                    if err is not None:
                        self.status.configure(text='')
                        messagebox.showerror(TITLE, str(err))
                    elif done:
                        done(res)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------ Ryujinx
    def detect(self):
        c = R.ryujinx_candidates()
        self.set_ryujinx(c[0] if c else '')

    def pick_ryujinx(self):
        d = filedialog.askdirectory(title='Choose the Ryujinx data folder (it contains "bis" and "mods")')
        if d:
            if os.path.isdir(os.path.join(d, 'portable')) and not os.path.isdir(os.path.join(d, 'bis')):
                d = os.path.join(d, 'portable')
            self.set_ryujinx(d)

    def set_ryujinx(self, ry):
        self.env = R.environment(ry)
        if self.env['problems']:
            self.env_lbl.configure(text='Ryujinx: ' + ' '.join(self.env['problems']), foreground='#b42318')
        else:
            where = 'mod folder "RbxToMii"'
            text = 'Ryujinx: %s  ✓  (%s)' % (ry, where)
            if self.env['notes']:
                text += '\n' + ' '.join(self.env['notes'])
            self.env_lbl.configure(text=text, foreground='#b54708' if self.env['notes'] else '#1a7f37')
        self.refresh_miis()
        self._update_add()

    # ------------------------------------------------------------ the island's Miis
    def _label(self, m):
        return '%s%s' % (m['name'], ('   (now: Roblox %s)' % m['roblox']) if m['roblox'] else '')

    def refresh_miis(self):
        """the list is read from the save every time it is opened (new Miis appear after the game saved them)"""
        old = self.selected_mii()
        try:
            self.miis = R.island_miis(self.env) if self.env and not self.env['problems'] else []
        except Exception:
            self.miis = []
        labels = [self._label(m) for m in self.miis]
        self.mii_box.configure(values=labels)
        keep = next((self._label(m) for m in self.miis if old and m['index'] == old['index']), None)
        self.mii_var.set(keep or NO_MII)

    def selected_mii(self):
        cur = self.mii_var.get()
        return next((m for m in self.miis if self._label(m) == cur), None)

    def suggest_mii(self, ch):
        """preselect: the Mii that already wears this character, else one named like it"""
        if self.selected_mii():
            return
        nick = R.mii_nickname(ch).lower()
        for m in self.miis:
            if m['roblox'] == ch['key'] or m['name'].lower() in (nick, ch['key'].lower()):
                self.mii_var.set(self._label(m))
                return

    # ------------------------------------------------------------ character
    def load_player(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showinfo(TITLE, 'Type a Roblox username first.')
            return
        self.work(lambda: self._load(lambda: R.fetch_player(name, log=self.say)), self.loaded)

    def load_file(self):
        p = filedialog.askopenfilename(title='Roblox file with the character (saved in Roblox Studio)',
                                       filetypes=[('Roblox', '*.rbxm *.rbxmx *.rbxl *.rbxlx'), ('All files', '*.*')])
        if not p:
            return
        models = R.rig_from_file(p)
        if not models:
            messagebox.showerror(TITLE, 'There is no complete R15 character (a model with a Humanoid) in this file.')
            return
        name = models[0][0]
        if len(models) > 1:
            name = self.choose('Which character?', [m[0] for m in models])
            if not name:
                return
        self.work(lambda: self._load(lambda: R.fetch_file(p, name, log=self.say)), self.loaded)

    def _load(self, fetch):
        ch = fetch()
        place = self.place_for(ch) or dict(R.templates()['places'][0], tag='Spot1')    # preview only
        return ch, R.build(ch, place, log=self.say)

    def loaded(self, val):
        ch, res = val
        self.ch = ch
        self.built = {res['place']['nr']: res}
        av = os.path.join(ch['dir'], 'avatar.png')
        if os.path.exists(av):
            a = Image.open(av).convert('RGBA')
            pic = Image.new('RGBA', a.size, (238, 241, 245, 255))
            pic.alpha_composite(a)
            self.show(self.pic_rbx, pic.convert('RGB'), 'rbx')
        self.show(self.pic_game, res['preview'][0], 'game')
        miss = [] if ch['head_real'] else ['head (a stand-in is used)']
        miss += [m['part'].replace(' (3D)', '').lower() for m in ch['missing']]
        if miss:
            bits = ['Missing: %s.' % ', '.join(dict.fromkeys(miss)),
                    'Roblox only gives these out when logged in. Show the character once in Roblox on this PC '
                    '(e.g. play a game together) and press Load again, or use Roblox Studio (see README.txt).']
        else:
            bits = ['✓ Complete, with the real Roblox head']
        self.info.configure(text='\n'.join(bits))
        self.refresh_miis()
        self.suggest_mii(ch)
        self.status.configure(text='Choose the Mii that should become %s, close Ryujinx, then press "Add to Tomodachi Life".'
                              % ch['key'])
        self._update_add()
        if os.environ.get('RBXTOMII_TEST_MII'):        # automatic run (tests): Mii by name
            want = os.environ['RBXTOMII_TEST_MII']
            m = next((m for m in self.miis if m['name'] == want), None)
            if m:
                self.mii_var.set(self._label(m))
                self.root.after(500, lambda: self.add(test=True))

    # ------------------------------------------------------------ add
    def place_for(self, ch):
        """the spot this character already has, else the first free one"""
        if not self.env or self.env['problems']:
            return None
        spots = R.places(self.env)
        same = [p for p in spots if p.get('roblox') == ch['key']]
        if same:
            return same[0]
        free = [p for p in spots if not p['now']]
        return free[0] if free else None

    def _update_add(self):
        ok = bool(self.ch) and self.env is not None and not self.env['problems'] and not self.busy
        self.bt_add.configure(state='normal' if ok else 'disabled')

    def choose(self, question, options):
        win = tk.Toplevel(self.root)
        win.title(TITLE)
        win.resizable(False, False)
        ttk.Label(win, text=question, padding=(14, 12, 14, 6), wraplength=360).pack(anchor='w')
        var = tk.StringVar(value=options[0])
        for o in options:
            ttk.Radiobutton(win, text=o, value=o, variable=var).pack(anchor='w', padx=24, pady=1)
        res = {}
        bf = ttk.Frame(win, padding=10)
        bf.pack(fill='x')
        ttk.Button(bf, text='Cancel', command=win.destroy).pack(side='right', padx=4)
        ttk.Button(bf, text='OK', command=lambda: (res.update(v=var.get()), win.destroy())).pack(side='right')
        win.transient(self.root)
        win.grab_set()
        self.root.wait_window(win)
        return res.get('v')

    def add(self, test=False):
        if not self.ch:
            return
        mii = self.selected_mii()
        if mii is None:
            messagebox.showinfo(TITLE, 'Choose the Mii that should become %s under "Mii in your island".\n\n'
                                       'No Mii for it yet? Make a new Mii in Tomodachi Life (any look), let the game save, '
                                       'close Ryujinx and choose it here.' % self.ch['key'])
            return
        if R.emulator_running():
            messagebox.showinfo(TITLE, 'Close Ryujinx first, then press the button again.\n\n'
                                       '(RbxToMii writes into your save. A running game would overwrite it.)')
            return
        place = self.place_for(self.ch)
        if place is None:
            spots = R.places(self.env)
            labels = ['%s (spot %d)' % (p['now'], p['nr']) for p in spots]
            pick = os.environ.get('RBXTOMII_TEST_REPLACE') if test else self.choose(
                'All 7 Roblox spots are taken. Which Roblox character should %s replace? '
                'Miis wearing that one will look like %s, too.' % (self.ch['key'], self.ch['key']), labels)
            if not pick:
                return
            place = next((p for p, l in zip(spots, labels) if l == pick or str(p['nr']) == pick), None)
            if place is None:
                return
        ch, env = self.ch, self.env

        def go():
            res = self.built.get(place['nr'])
            if res is None or res['place'].get('tag') != place['tag']:      # built for another spot / file names
                res = R.build(ch, place, log=self.say)
            self.built[place['nr']] = res
            self.say('Adding to the game …')
            return R.install(res, ch, env, mii=mii['index'], log=self.say)

        def done(r):
            self.refresh_miis()
            self.status.configure(text='Done: %s is now %s (backup: %s).' % (r['mii'], ch['key'], os.path.basename(r['backup'])))
            if test:
                return
            extra = ('The Roblox items are in the mod folder "RbxToMii" (Ryujinx: right-click the game → '
                     'Open Mods Directory). It has to stay enabled under Manage Mods.')
            messagebox.showinfo(TITLE, 'Done! %s now looks like %s.\n\nStart Tomodachi Life.\n\n%s' % (r['mii'], ch['key'], extra))
        self.work(go, done)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista')
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
