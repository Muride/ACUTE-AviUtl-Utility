import os
import sys
import subprocess
import re
import shutil
import tkinter
from tkinter import filedialog, messagebox
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

HEADER_TEMPLATE = """[project]
file={filename}
display.scene=0
[scene.0]
scene=0
name=Root
video.width=1920
video.height=1080
video.rate={fps}
video.scale=1
audio.rate=44100
cursor.frame=0
display.frame=0
display.layer=0
display.zoom=3000
"""

class AviUtlGenerator:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback

    def print_log(self, text):
        print(text)
        if self.log_callback:
            self.log_callback(text)

    def sec_to_frame(self, sec, fps):
        return max(1, int(round(sec * fps)))

    def detect_silence(self, file_path, threshold_db, min_silence_sec):
        cmd = [
            "ffmpeg", "-i", file_path,
            "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence_sec}",
            "-f", "null", "-"
        ]
        self.print_log(f"無音解析中: {os.path.basename(file_path)} ...")
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, encoding='cp932', errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW)
        
        silence_parts = []
        start_time = None
        for line in result.stderr.splitlines():
            if "silence_start" in line:
                m = re.search(r"silence_start: ([\d\.]+)", line)
                if m: start_time = float(m.group(1))
            elif "silence_end" in line:
                m = re.search(r"silence_end: ([\d\.]+)", line)
                if m and start_time is not None:
                    silence_parts.append((start_time, float(m.group(1))))
                    start_time = None
        
        duration = self.get_duration(file_path)
        keep_parts = []
        cursor = 0.0
        for s, e in silence_parts:
            if s > cursor: keep_parts.append((cursor, s))
            cursor = e
        if cursor < duration: keep_parts.append((cursor, duration))
        return keep_parts

    def apply_margin_and_gap(self, keep_parts, margin_s, min_gap_s, total_duration):
        if not keep_parts: return []
        expanded = [(max(0, s - margin_s), min(total_duration, e + margin_s)) for s, e in keep_parts]
        expanded.sort(key=lambda x: x[0])
        
        merged = []
        c_s, c_e = expanded[0]
        for n_s, n_e in expanded[1:]:
            if n_s <= c_e + 0.001:
                c_e = max(c_e, n_e)
            else:
                merged.append((c_s, c_e))
                c_s, c_e = n_s, n_e
        merged.append((c_s, c_e))

        final_parts = []
        c_s, c_e = merged[0]
        for n_s, n_e in merged[1:]:
            if (n_s - c_e) < min_gap_s:
                c_e = n_e
            else:
                final_parts.append((c_s, c_e))
                c_s, c_e = n_s, n_e
        final_parts.append((c_s, c_e))
        return final_parts

    def get_duration(self, file_path):
        cmd = ["ffmpeg", "-i", file_path, "2>&1"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='cp932', errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW)
        for line in res.stderr.splitlines():
            if "Duration" in line:
                t = re.search(r"Duration: (\d+):(\d+):([\d\.]+)", line)
                if t: return float(t.group(1))*3600 + float(t.group(2))*60 + float(t.group(3))
        return 0.0

    def export_video(self, source_file, keep_parts, output_file):
        temp_dir = "temp_jetcut_video"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        list_file_path = os.path.join(temp_dir, "concat_list.txt")
        ext = os.path.splitext(source_file)[1]
        
        try:
            with open(list_file_path, "w", encoding="utf-8") as f:
                for i, (s, e) in enumerate(keep_parts):
                    chunk_path = os.path.abspath(os.path.join(temp_dir, f"chunk_{i:04d}{ext}"))
                    cmd = ["ffmpeg", "-y", "-i", source_file, "-ss", str(s), "-to", str(e), "-c", "copy", chunk_path]
                    self.print_log(f"[DEBUG] 動画カット {i+1}/{len(keep_parts)}...")
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
                    f.write(f"file '{chunk_path.replace('\\', '/')}'\n")

            self.print_log("動画を結合中...")
            cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", output_file]
            subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        finally:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

    def create_aup_object(self, idx, layer, start_f, end_f, file_path, seek_start, seek_end):
        safe_path = file_path.replace("\\", "/").replace("/", "\\")
        ext = os.path.splitext(file_path)[1].lower()
        is_audio = ext in ['.wav', '.mp3', '.m4a', '.aac', '.flac']
        obj_text = f"[{idx}]\nlayer={layer}\nframe={start_f},{end_f}\n"
        if is_audio:
            obj_text += (f"[{idx}.0]\neffect.name=音声ファイル\n再生位置={seek_start:.2f},{seek_end:.2f},再生範囲,0\n"
                         f"再生速度=100.00\nファイル={safe_path}\nトラック=0\nループ再生=0\n"
                         f"[{idx}.1]\neffect.name=音声再生\n音量=100.00\n左右=0.00")
        else:
            obj_text += (f"[{idx}.0]\neffect.name=動画ファイル\n再生位置={seek_start:.2f},{seek_end:.2f},再生範囲,0\n"
                         f"再生速度=100.00\nループ再生=0\nアルファチャンネルを読み込む=0\nファイル={safe_path}\n音声付き=1\n"
                         f"[{idx}.1]\neffect.name=映像再生\nX=0.0\nY=0.0\nZ=0.0\n拡大率=100.00\n透明度=0.0\n合成モード=通常\n音量=100.0")
        return obj_text


class AppJetcutter(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("ACUTE Jetcutter v1.1.2")
        self.geometry("900x700")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # 左カラム
        self.frame_left = ctk.CTkFrame(self)
        self.frame_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # モード選択
        self.mode_var = ctk.StringVar(value="aup2")
        frame_mode = ctk.CTkFrame(self.frame_left)
        frame_mode.pack(fill="x", pady=10, padx=10)
        ctk.CTkLabel(frame_mode, text="出力モード:").pack(side="left", padx=5)
        ctk.CTkRadioButton(frame_mode, text="AUP2出力", variable=self.mode_var, value="aup2", command=self.on_mode_change).pack(side="left", padx=10)
        ctk.CTkRadioButton(frame_mode, text="動画出力 (単一ファイル)", variable=self.mode_var, value="video", command=self.on_mode_change).pack(side="left", padx=10)

        # ファイル
        self.entry_base = self.create_dnd_entry(self.frame_left, "【必須】ベース動画:")
        ctk.CTkLabel(self.frame_left, text="同期ファイル (AUP2モードのみ):").pack(anchor="w", padx=10, pady=(5,0))
        self.textbox_sync = ctk.CTkTextbox(self.frame_left, height=60)
        self.textbox_sync.pack(fill="x", padx=10, pady=5)
        self.textbox_sync._textbox.drop_target_register(DND_FILES)
        self.textbox_sync._textbox.dnd_bind('<<Drop>>', self.drop_sync_files)

        # 設定
        frame_set = ctk.CTkFrame(self.frame_left)
        frame_set.pack(fill="x", padx=10, pady=10)
        self.spin_th = self.create_setting(frame_set, "閾値(dB):", "-30", 0)
        self.spin_dur = self.create_setting(frame_set, "無音秒数(s):", "0.3", 2)
        self.spin_margin = self.create_setting(frame_set, "余白(f):", "5", 4)
        
        self.chk_split_only = ctk.CTkCheckBox(self.frame_left, text="無音部分の分割のみ (削除しない)")
        self.chk_split_only.pack(anchor="w", padx=20, pady=5)

        self.entry_output = self.create_dnd_entry(self.frame_left, "出力ファイル (任意):", placeholder="指定なしなら自動生成")

        ctk.CTkButton(self.frame_left, text="処理開始", height=50, command=self.run_process).pack(fill="x", padx=10, pady=20)

        # 右カラム (ログ)
        self.textbox_log = ctk.CTkTextbox(self, width=350)
        self.textbox_log.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.generator = AviUtlGenerator(log_callback=self.log)

    def create_dnd_entry(self, parent, label_text, placeholder=""):
        ctk.CTkLabel(parent, text=label_text).pack(anchor="w", padx=10)
        frm = ctk.CTkFrame(parent, fg_color="transparent")
        frm.pack(fill="x", padx=10, pady=2)
        entry = ctk.CTkEntry(frm, placeholder_text=placeholder)
        entry.pack(side="left", fill="x", expand=True)
        entry._entry.drop_target_register(DND_FILES)
        entry._entry.dnd_bind('<<Drop>>', lambda e: self.on_drop(e, entry))
        return entry

    def create_setting(self, parent, label, default, col):
        ctk.CTkLabel(parent, text=label).grid(row=0, column=col, padx=5)
        entry = ctk.CTkEntry(parent, width=50)
        entry.insert(0, default)
        entry.grid(row=0, column=col+1)
        return entry

    def on_drop(self, event, entry):
        path = event.data.strip("{}")
        entry.delete(0, "end")
        entry.insert(0, path)

    def drop_sync_files(self, event):
        files = event.data.replace("}{", "\n").replace("{", "").replace("}", "")
        self.textbox_sync.insert("end", files + "\n")

    def on_mode_change(self):
        if self.mode_var.get() == "video":
            self.textbox_sync.configure(state="disabled")
            self.chk_split_only.configure(state="disabled")
        else:
            self.textbox_sync.configure(state="normal")
            self.chk_split_only.configure(state="normal")

    def log(self, text):
        self.textbox_log.insert("end", text + "\n")
        self.textbox_log.see("end")
        self.update()

    def run_process(self):
        base_file = self.entry_base.get().strip()
        if not os.path.exists(base_file):
            messagebox.showerror("エラー", "ベース動画が見つかりません。")
            return

        out_path = self.entry_output.get().strip()
        fps = 30.0
        mode = self.mode_var.get()
        split_only = bool(self.chk_split_only.get()) and (mode == "aup2")
        
        self.log(f"=== JetCut 開始 ({mode} モード) ===")
        dur = self.generator.get_duration(base_file)
        raw_keep = self.generator.detect_silence(base_file, float(self.spin_th.get()), float(self.spin_dur.get()))
        keep_parts = self.generator.apply_margin_and_gap(raw_keep, int(self.spin_margin.get())/fps, 0.2, dur)

        if mode == "video":
            if not out_path:
                name, ext = os.path.splitext(base_file)
                out_path = f"{name}_jetcut{ext}"
            self.generator.export_video(base_file, keep_parts, out_path)
            
        else: # AUP2 Mode
            if not out_path: out_path = os.path.splitext(base_file)[0] + "_jetcut.aup2"
            sync_files = [f.strip() for f in self.textbox_sync.get("1.0", "end").split('\n') if f.strip()]
            targets = [base_file] + sync_files

            timeline = []
            if split_only:
                cursor = 0.0
                for s, e in keep_parts:
                    if s > cursor: timeline.append((cursor, s)) # 無音部分も残す
                    timeline.append((s, e))
                    cursor = e
                if cursor < dur: timeline.append((cursor, dur))
            else:
                timeline = keep_parts

            body = []
            obj_idx = 0
            prev_end_f = -1

            for s, e in timeline:
                start_f = prev_end_f + 1
                end_f = start_f + self.generator.sec_to_frame(e - s, fps) - 1
                for i, fpath in enumerate(targets):
                    body.append(self.generator.create_aup_object(obj_idx, i+1, start_f, end_f, fpath, s, e))
                    obj_idx += 1
                prev_end_f = end_f

            header = HEADER_TEMPLATE.format(filename="generated.aup2", fps=fps)
            for i in range(len(targets)): header += f"[layer.{i}]\nlayer={i+1}\nname=Layer{i+1}\n"
            
            with open(out_path, "wb") as f:
                f.write((header + "\n".join(body)).encode('utf-8'))

        self.log("完了しました！")

if __name__ == "__main__":
    app = AppJetcutter()
    app.mainloop()