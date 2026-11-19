"""Full-resolution desktop view of FoundationPose inference frames."""

import multiprocessing as mp
from queue import Empty, Full
import time

import cv2


def _display(title, updates, stopped, stats):
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"FoundationPose | {title}")
    result = tk.Label(root, text="Waiting for pose", compound="top")
    result.pack()
    root.protocol("WM_DELETE_WINDOW", stopped.set)
    for key in ("<Escape>", "<q>", "<Q>"):
        root.bind(key, lambda _: stopped.set())

    def show(label, rgb):
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=photo)
        label.image = photo

    first = None
    try:
        while not stopped.is_set():
            root.update()
            try:
                status, rgb = updates.get_nowait()
            except Empty:
                continue
            show(result, rgb)
            result.configure(text=status)
            now = time.monotonic()
            if first is None:
                first = now
            with stats.get_lock():
                stats[0] += 1
                stats[1] = now - first
    finally:
        root.destroy()


class GeminiPreview:
    def __init__(self, title):
        ctx = mp.get_context('spawn')  # Do not fork the initialized CUDA runtime.
        self.updates = ctx.Queue(maxsize=1)
        self.stop = ctx.Event()
        self.stats = ctx.Array('d', 2)
        self.process = ctx.Process(target=_display, daemon=True,
                                   args=(title, self.updates, self.stop, self.stats))
        self.process.start()
        self.source = None

    @property
    def closed(self):
        if self.process.exitcode not in (None, 0):
            raise RuntimeError(f'Gemini preview exited with status {self.process.exitcode}')
        return self.stop.is_set()

    def present(self, bgr, status):
        if self.closed:
            return False
        number, timestamp = self.source
        caption = f'{status}\nSource frame {number} | ROS {timestamp / 1000:.3f} s'
        try:
            self.updates.put_nowait((caption, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        except Full:
            pass  # Never block inference on a busy display.
        return True

    def measurements(self):
        with self.stats.get_lock():
            count, seconds = self.stats[:]
        return {'preview_frames': int(count), 'preview_duration_s': seconds,
                'preview_fps': (count - 1) / seconds if seconds > 0 else None}

    def close(self):
        self.stop.set()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.updates.cancel_join_thread()
        self.updates.close()
