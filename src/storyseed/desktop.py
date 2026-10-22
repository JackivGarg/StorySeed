"""A local Tk interface; all widgets stay on the Tk thread."""
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from storyseed.app_service import GenerationRequest, GenerationService


class StorySeedWindow:
    def __init__(self, root, service_factory=GenerationService):
        self.root = root
        self.service_factory = service_factory
        self.service = None
        self.last_result = None
        self.events = queue.Queue()
        self.busy = True
        self.status = tk.StringVar(value='Loading the trained model...')
        root.title('StorySeed | A small beginning, a new story')
        root.geometry('1000x790')
        root.minsize(760, 680)
        root.configure(bg='#101c2a')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background='#101c2a')
        style.configure('TLabel', background='#101c2a', foreground='#e9eef3', font=('Segoe UI', 11))
        style.configure('Title.TLabel', font=('Segoe UI', 27, 'bold'), foreground='#a7edca')
        style.configure('TButton', font=('Segoe UI', 11), padding=(14, 8))
        frame = ttk.Frame(root, padding=24)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='StorySeed', style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame, text='Write an opening. Explore what our scratch-trained model continues.').pack(anchor='w', pady=(3, 14))
        ttk.Label(frame, text='Experimental model: stories can repeat or lose track of characters.').pack(anchor='w', pady=(0, 10))
        ttk.Label(frame, text='YOUR STORY OPENING').pack(anchor='w')
        self.prompt = ScrolledText(frame, height=4, wrap='word', font=('Segoe UI', 12), bg='#f4f6f2', fg='#182332', padx=12, pady=10)
        self.prompt.pack(fill='x', pady=(6, 12))
        self.prompt.insert('1.0', 'Once upon a time')
        options = ttk.Frame(frame)
        options.pack(fill='x', pady=(0, 12))
        self.settings = {}
        for column, (label, value) in enumerate([('Seed', '17'), ('Temperature', '0.8'), ('Top-k', '40'), ('New IDs', '192')]):
            cell = ttk.Frame(options)
            cell.grid(row=0, column=column, padx=(0, 18), sticky='w')
            ttk.Label(cell, text=label).pack(anchor='w')
            variable = tk.StringVar(value=value)
            ttk.Entry(cell, textvariable=variable, width=12).pack(anchor='w', pady=4)
            self.settings[label] = variable
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x', pady=(0, 12))
        self.generate_button = ttk.Button(buttons, text='Continue story', command=self.submit, state='disabled')
        self.generate_button.pack(side='left')
        self.copy_button = ttk.Button(buttons, text='Copy story', command=self.copy, state='disabled')
        self.copy_button.pack(side='left', padx=10)
        self.save_button = ttk.Button(buttons, text='Save result', command=self.save, state='disabled')
        self.save_button.pack(side='left')
        ttk.Label(frame, text='CONTINUATION').pack(anchor='w')
        self.output = ScrolledText(frame, height=12, wrap='word', font=('Segoe UI', 12), bg='#eaf4ee', fg='#182332', padx=12, pady=12, state='disabled')
        self.output.pack(fill='both', expand=True, pady=(6, 10))
        ttk.Label(frame, textvariable=self.status, wraplength=900).pack(anchor='w')
        self.footer = ttk.Label(frame, text='Learning prototype · loading model identity...\nGenerated text can still contain errors. New IDs are not words.',
                  wraplength=900, foreground='#afbccb')
        self.footer.pack(anchor='w', pady=(12, 0))
        threading.Thread(target=self._load, daemon=True).start()
        root.after(80, self.poll)

    def _load(self):
        try:
            self.events.put(('loaded', self.service_factory()))
        except Exception as error:
            self.events.put(('load_error', str(error)))

    def _generate(self, request):
        try:
            self.events.put(('result', self.service.generate(request)))
        except Exception as error:
            self.events.put(('generation_error', str(error)))

    def submit(self):
        if self.busy or self.service is None:
            return
        try:
            request = GenerationRequest(self.prompt.get('1.0', 'end-1c'), int(self.settings['Seed'].get()),
                                        float(self.settings['Temperature'].get()), int(self.settings['Top-k'].get()),
                                        int(self.settings['New IDs'].get()))
            request.validate()
        except (ValueError, TypeError) as error:
            messagebox.showerror('Check generation settings', str(error), parent=self.root)
            return
        self.busy = True
        self.generate_button.configure(state='disabled')
        self.status.set('Generating locally... the window remains available.')
        threading.Thread(target=self._generate, args=(request,), daemon=True).start()

    def poll(self):
        try:
            while True:
                event, value = self.events.get_nowait()
                self.busy = False
                if event == 'loaded':
                    self.service = value
                    self.footer.configure(text=f"{value.metadata['model']} · {value.metadata['parameters']:,} parameters · {value.metadata['context']}-ID context · local CPU\nLearning prototype; text can contain errors. New IDs are not words.")
                    self.generate_button.configure(state='normal')
                    self.status.set('Ready. The trained model stays fixed while you explore continuations.')
                elif event == 'result':
                    self.last_result = value
                    self.output.configure(state='normal')
                    self.output.delete('1.0', 'end')
                    self.output.insert('1.0', value['continuation'])
                    self.output.configure(state='disabled')
                    for button in (self.generate_button, self.copy_button, self.save_button):
                        button.configure(state='normal')
                    self.status.set(f"{len(value['generated_ids'])} new IDs · stop: {value['stop_reason']} · {value['elapsed_seconds']:.2f}s · "
                                    f"initial prompt: {value['initial_visible_ids']} visible, {value['initial_cropped_ids']} cropped IDs")
                else:
                    if self.service is not None:
                        self.generate_button.configure(state='normal')
                    self.status.set('Could not load model. Close and reopen after fixing setup.' if event == 'load_error' else 'Generation failed; edit settings and try again.')
                    messagebox.showerror('StorySeed', value, parent=self.root)
        except queue.Empty:
            pass
        self.root.after(80, self.poll)

    def copy(self):
        if self.last_result:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_result['text'])

    def save(self):
        if not self.last_result:
            return
        filename = filedialog.asksaveasfilename(parent=self.root, title='Save story and generation settings',
                                               defaultextension='.json', initialfile='storyseed-result.json',
                                               filetypes=[('StorySeed result', '*.json')])
        if filename:
            try:
                Path(filename).write_text(json.dumps(self.last_result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            except OSError as error:
                messagebox.showerror('Could not save', str(error), parent=self.root)


def launch(service_factory=GenerationService):
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(f'Tcl/Tk could not create a window: {error}') from error
    StorySeedWindow(root, service_factory)
    root.mainloop()
