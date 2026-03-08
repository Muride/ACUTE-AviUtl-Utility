import os
# --- 追加: AIのサイレントクラッシュ（OpenMP競合）をOSレベルで強制防止 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
# -------------------------------------------------------------------
import sys
import subprocess
import re
import math
import shutil
import json
import tkinter
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox, font, colorchooser
import customtkinter as ctk
import torch
from tkinterdnd2 import DND_FILES, TkinterDnD
import MeCab
import ipadic
from faster_whisper import WhisperModel

# Google GenAI SDK
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# =========================================================
# 設定・定数
# =========================================================
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

LINE_CHARACTER_LIMIT = 15
GEMINI_BATCH_SIZE = 20

# Geminiモデルのプリセット
GEMINI_PRESETS = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash"
]

# Whisperモデルの選択肢
WHISPER_MODELS = [
    "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
]

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

# =========================================================
# ロジッククラス (文字起こし特化)
# =========================================================

class AviUtlGenerator:
    def __init__(self, log_callback=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log_callback = log_callback
        
        self.print_log(f"使用デバイス: {self.device}")
        
        if not GENAI_AVAILABLE:
            self.print_log("警告: 'google-genai' ライブラリが見つかりません。")
        
        try:
            self.print_log("MeCabを初期化しています...")
            self.tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
        except Exception as e:
            self.print_log(f"MeCab(ipadic)の初期化に失敗しました: {e}")
            try:
                self.tagger = MeCab.Tagger()
            except Exception as e2:
                self.print_log(f"MeCab初期化失敗: {e2}")
                self.tagger = None

    def print_log(self, text):
        print(text)
        if self.log_callback:
            self.log_callback(text)

    def sec_to_frame(self, sec, fps):
        f = int(round(sec * fps))
        return max(1, f)

    def get_duration(self, file_path):
        cmd = ["ffmpeg", "-i", file_path, "2>&1"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='cp932', errors='ignore', creationflags=subprocess.CREATE_NO_WINDOW)
        for line in res.stderr.splitlines():
            if "Duration" in line:
                t = re.search(r"Duration: (\d+):(\d+):([\d\.]+)", line)
                if t: return float(t.group(1))*3600 + float(t.group(2))*60 + float(t.group(3))
        return 0.0

    def get_mecab_chunks(self, text):
        if self.tagger is None: return [text]
        if len(text) > 300:
            self.print_log(f"[DEBUG] 異常な長文({len(text)}文字)を検知！MeCabをスキップします。")
            return [text]
        node = self.tagger.parseToNode(text)
        chunks = []
        current_chunk = ""
        independent_pos = ["名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭詞"]
        last_pos = ""
        while node:
            word = node.surface
            if not word:
                node = node.next
                continue
            features = node.feature.split(",")
            pos = features[0]
            pos_sub = features[1]
            if current_chunk:
                is_independent = (pos in independent_pos)
                is_suffix = (pos_sub == "接尾") or ("非自立" in features) or (pos == "助詞") or (pos == "助動詞") or (pos == "記号")
                prev_is_prefix = (last_pos == "接頭詞")
                if is_independent and not is_suffix and not prev_is_prefix:
                    chunks.append(current_chunk)
                    current_chunk = ""
            current_chunk += word
            last_pos = pos
            node = node.next
        if current_chunk: chunks.append(current_chunk)
        return chunks

    def parse_alias(self, file_path):
        """ .objectファイルを解析し、エフェクトブロックのリストを返す """
        blocks = []
        current_lines = []
        content = ""
        try:
            with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="cp932", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                raise Exception(f"エイリアスファイルの読み込みに失敗しました: {e}")

        lines = content.splitlines()
        capture = False
        
        for line in lines:
            line = line.strip()
            if not line: continue
            if re.match(r'^\[Object\.\d+\]$', line):
                if current_lines:
                    blocks.append(current_lines)
                current_lines = []
                capture = True
                continue
            if capture:
                current_lines.append(line)
        if current_lines:
            blocks.append(current_lines)
        return blocks

    def create_subtitle_object_alias(self, idx, start_f, end_f, text, alias_blocks, layer_id=2):
        res = f"[{idx}]\nlayer={layer_id}\nframe={start_f},{end_f}\n"
        for i, block in enumerate(alias_blocks):
            res += f"[{idx}.{i}]\n"
            for line in block:
                if line.startswith("テキスト="):
                    res += f"テキスト={text}\n"
                else:
                    res += line + "\n"
        return res

    def refine_subtitles_with_gemini(self, original_lines, api_key, user_context, keywords, debug_log_path, is_debug, target_model_name):
        if not GENAI_AVAILABLE:
            self.print_log("google-genaiライブラリがないためスキップします。")
            return original_lines
        if not api_key:
            self.print_log("APIキーがないためスキップします。")
            return original_lines

        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            self.print_log(f"Gemini Client初期化エラー: {e}")
            return original_lines

        self.working_model_name = target_model_name
        self.print_log(f"Gemini({self.working_model_name})による字幕修正を開始します...")
        
        f_log = None
        if is_debug:
            self.print_log(f"デバッグログを出力します: {debug_log_path}")
            f_log = open(debug_log_path, "w", encoding="utf-8")
            f_log.write(f"CONTEXT: {user_context}\nKEYWORDS: {keywords}\nMODEL: {self.working_model_name}\n\n")
        
        corrected_lines = []
        
        try:
            for i in range(0, len(original_lines), GEMINI_BATCH_SIZE):
                batch = original_lines[i : i + GEMINI_BATCH_SIZE]
                self.print_log(f"  - Batch Processing: {i+1} ~ {min(i+GEMINI_BATCH_SIZE, len(original_lines))}...")
                
                if f_log: f_log.write(f"--- BATCH {i} REQUEST ---\n")

                input_data = []
                for idx, item in enumerate(batch):
                    input_data.append({
                        "id": idx, 
                        "text": item['text'], 
                        "start": round(item['start'], 3),
                        "end": round(item['end'], 3)
                    })
                
                prompt = f"""
あなたは動画字幕の校正AIです。入力されたJSONデータの字幕テキストを修正し、JSON形式で出力してください。

【コンテキスト】
動画の内容: {user_context}
キーワード: {keywords}

【タスクと指示】
1. **誤変換の修正**: 
   - 音声認識特有の誤字（同音異義語、漢字の変換ミス）を文脈から判断して修正してください。
   - 例: "後世"→"構成", "イッタ"→"行った", "都市濃し"→"年越し"
   - 指定された「キーワード」は最優先で適用してください。

2. **不自然な区切りの調整**:
   - 基本的に入力の分割（MeCab処理済み）を尊重してください。
   - ただし、文節の途中など明らかに不自然な切れ方をしている場合（例: 「お使いの環」「境では、ソフトを～」）は、文の流れに従って切れる場所を修正してください。
   - 1行が極端に短い/長い場合は調整して構いませんが、目安は15文字程度です。

3. **タイムスタンプの厳守（最重要）**:
   - 入力の `start` と `end` は、音声の絶対時間です。**原則として値を変更しないでください。**
   - 以前の行の時間をずらしたり、積み上げ計算をしないでください。
   - **例外**: 文節の途中など明らかに不自然な切れ方をしている場合の修正を行った結果、行の文字数が変わる場合は、文字数比率などで各行の `start`~`end` の間を案分して推定し、`start`~`end`の値を修正してください。
   - **例外**: 行を「分割」する場合は、文字数比率などで `start`~`end` の間を案分して推定してください。

【出力形式】
JSONの配列(list of objects)で返してください。
各オブジェクトは `{{ "text": "修正後テキスト", "start": 10.5, "end": 12.0 }}` の形式です。

【入力データ】
{json.dumps(input_data, ensure_ascii=False)}

【出力JSON】
```json
"""
                if f_log: f_log.write(prompt + "\n")

                try:
                    response = client.models.generate_content(
                        model=self.working_model_name,
                        contents=prompt
                    )
                    res_text = response.text
                    
                    if f_log:
                        f_log.write(f"--- BATCH {i} RESPONSE ---\n")
                        f_log.write(res_text + "\n\n")

                    res_text = re.sub(r"```json", "", res_text)
                    res_text = re.sub(r"```", "", res_text).strip()
                    corrected_batch = json.loads(res_text)
                    
                    for item in corrected_batch:
                        start_t = float(item.get("start", 0.0))
                        end_t = float(item.get("end", 0.0))
                        txt = item.get("text", "")
                        
                        corrected_lines.append({
                            "text": txt,
                            "start": start_t,
                            "end": end_t
                        })
                        
                except Exception as e:
                    self.print_log(f"  ! Batch Error: {e}")
                    if f_log: f_log.write(f"--- BATCH {i} ERROR ---\n{e}\n\n")
                    corrected_lines.extend(batch)
        finally:
            if f_log: f_log.close()

        return corrected_lines

    def transcribe_segment_based(self, file_path, base_file_path, model_size="medium", remove_filler=True, 
                                 use_gemini=False, gemini_key="", gemini_context="", gemini_keywords="", is_debug=False,
                                 target_gemini_model="gemini-1.5-flash-latest"):
        
        if not hasattr(self, "whisper_model") or getattr(self, "current_model_size", "") != model_size:
            self.print_log(f"Faster-Whisperモデル({model_size})ロード中...")
            self.whisper_model = WhisperModel(model_size, device=self.device, compute_type="float16" if self.device=="cuda" else "int8")
            self.current_model_size = model_size
        else:
            self.print_log(f"Faster-Whisperモデル({model_size})はロード済みです。再利用します。")
            
        model = self.whisper_model
        
        self.print_log("文字起こし実行中...")
        segments, info = model.transcribe(
            file_path, beam_size=5, word_timestamps=True, language="ja",
            vad_filter=True, condition_on_previous_text=False
        )
        
        fillers = ["えー", "あの", "その", "えっと", "あー", "んー", "まぁ", "うーんと", "あ、", "ん、"]
        raw_lines = []
        
        for segment in segments:
            # self.print_log(f"[DEBUG] セグメント解析... 長さ: {len(segment.text)}文字 | テキスト: {segment.text[:30]}")
            seg_text_clean = ""
            char_time_map = []
            if not segment.words: continue
            for w in segment.words:
                word_text = w.word
                clean_w = re.sub(r"[、。！？]", "", word_text).strip()
                if remove_filler and clean_w in fillers: continue
                duration = w.end - w.start
                length = len(word_text)
                if length == 0: continue
                char_dur = duration / length
                for i in range(length):
                    char_time_map.append({
                        "char": word_text[i],
                        "start": w.start + (i * char_dur),
                        "end": w.start + ((i+1) * char_dur)
                    })
                    seg_text_clean += word_text[i]
            
            if not seg_text_clean: continue

            limit = LINE_CHARACTER_LIMIT
            if len(seg_text_clean) <= limit:
                raw_lines.append({
                    "text": seg_text_clean,
                    "start": char_time_map[0]['start'],
                    "end": char_time_map[-1]['end']
                })
            else:
                chunks = self.get_mecab_chunks(seg_text_clean)
                global_char_idx = 0
                current_line_str = ""
                for chunk in chunks:
                    if len(current_line_str) + len(chunk) > limit:
                        if current_line_str:
                            s_idx = global_char_idx - len(current_line_str)
                            e_idx = global_char_idx - 1
                            if s_idx < len(char_time_map) and e_idx < len(char_time_map):
                                raw_lines.append({
                                    "text": current_line_str,
                                    "start": char_time_map[s_idx]['start'],
                                    "end": char_time_map[e_idx]['end']
                                })
                            current_line_str = ""
                    current_line_str += chunk
                    global_char_idx += len(chunk)
                if current_line_str:
                    s_idx = global_char_idx - len(current_line_str)
                    e_idx = global_char_idx - 1
                    if s_idx < len(char_time_map) and e_idx < len(char_time_map):
                        raw_lines.append({
                            "text": current_line_str,
                            "start": char_time_map[s_idx]['start'],
                            "end": char_time_map[e_idx]['end']
                        })

        if use_gemini and gemini_key:
            debug_log_path = base_file_path + "_gemini_debug_log.txt"
            final_lines = self.refine_subtitles_with_gemini(raw_lines, gemini_key, gemini_context, gemini_keywords, debug_log_path, is_debug, target_gemini_model)
        else:
            final_lines = raw_lines

        return final_lines

    def create_base_object(self, idx, layer, start_f, end_f, file_path):
        safe_path = file_path.replace("\\", "/").replace("/", "\\")
        ext = os.path.splitext(file_path)[1].lower()
        is_audio = ext in ['.wav', '.mp3', '.m4a', '.aac', '.flac']
        obj_text = f"[{idx}]\nlayer={layer}\nframe={start_f},{end_f}\n"
        if is_audio:
            obj_text += (f"[{idx}.0]\neffect.name=音声ファイル\n"
                         f"再生位置=0.00,0.00,再生範囲,0\n"
                         f"再生速度=100.00\nファイル={safe_path}\n"
                         f"トラック=0\nループ再生=0\n"
                         f"[{idx}.1]\neffect.name=音声再生\n音量=100.00\n左右=0.00")
        else:
            obj_text += (f"[{idx}.0]\neffect.name=動画ファイル\n"
                         f"再生位置=0.00,0.00,再生範囲,0\n"
                         f"再生速度=100.00\nループ再生=0\nアルファチャンネルを読み込む=0\n"
                         f"ファイル={safe_path}\n音声付き=1\n"
                         f"[{idx}.1]\neffect.name=映像再生\nX=0.0\nY=0.0\nZ=0.0\n拡大率=100.00\n透明度=0.0\n合成モード=通常\n音量=100.0")
        return obj_text

    def create_subtitle_object(self, idx, start_f, end_f, text, style_config, layer_id=2):
        font = style_config.get('font', 'Noto Sans JP Black')
        size = style_config.get('size', 70)
        color = style_config.get('color', 'e66c1a')
        outline_color = style_config.get('outline_color', 'ffffff')
        
        return (f"[{idx}]\nlayer={layer_id}\nframe={start_f},{end_f}\n"
                f"[{idx}.0]\neffect.name=テキスト\n"
                f"サイズ={size:.2f}\n字間=0.00\n行間=0.00\n表示速度=0.00\n"
                f"フォント={font}\n文字色={color}\n影・縁色=000000\n"
                f"文字装飾=標準文字\n文字揃え=中央揃え[下]\nB=0\nI=0\n"
                f"テキスト={text}\n文字毎に個別オブジェクト=0\n"
                f"移動座標上に表示=0\nオブジェクトの長さを自動調節=0\n"
                f"[{idx}.1]\neffect.name=標準描画\n"
                f"X=0.00\nY=510.00\nZ=0.00\n拡大率=100.000\n透明度=0.00\n"
                f"[{idx}.2]\neffect.name=縁取り\nサイズ=8\nぼかし=5\n縁色={outline_color}\nパターン画像=")

    def format_srt_time(self, sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"


# =========================================================
# GUIクラス
# =========================================================

class AppScripter(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("ACUTE Scripter v1.1.2 - Text & Subtitle Generator")
        self.geometry("1200x800")
        
        try:
            if getattr(sys, 'frozen', False):
                self.iconbitmap(sys.executable)
            elif os.path.exists("icon.ico"):
                self.iconbitmap("icon.ico")
        except:
            pass

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.ph_context = "動画の内容を入力します。例: AviUtlの解説動画。"
        self.ph_keywords = "文字起こしで間違えやすい固有名詞をカンマ区切りで入力します。例: AviUtl, 拡張編集"

        # --- 左カラム ---
        self.frame_left = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.frame_left.grid_columnconfigure(0, weight=1)

        # 1. 出力モード選択
        self.mode_var = ctk.StringVar(value="srt")
        frm_mode = ctk.CTkFrame(self.frame_left)
        frm_mode.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(frm_mode, text="出力モード:").pack(side="left", padx=10)
        ctk.CTkRadioButton(frm_mode, text="SRT出力", variable=self.mode_var, value="srt", command=self.on_mode_change).pack(side="left", padx=10)
        ctk.CTkRadioButton(frm_mode, text="AUP2出力", variable=self.mode_var, value="aup2", command=self.on_mode_change).pack(side="left", padx=10)

        # 2. ファイル選択
        self.frame_file = ctk.CTkFrame(self.frame_left)
        self.frame_file.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.frame_file, text="【必須】動画/音声ファイル:").pack(anchor="w", padx=5)
        frm_base = ctk.CTkFrame(self.frame_file, fg_color="transparent")
        frm_base.pack(fill="x", padx=5, pady=2)
        self.entry_base = ctk.CTkEntry(frm_base)
        self.entry_base.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.make_dnd(self.entry_base)
        ctk.CTkButton(frm_base, text="参照", width=60, command=lambda: self.browse(self.entry_base)).pack(side="right")

        # 3. 字幕設定 (タブ化)
        self.frame_sub = ctk.CTkFrame(self.frame_left)
        self.frame_sub.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(self.frame_sub, text="■ 字幕設定", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.chk_remove_filler = ctk.CTkCheckBox(self.frame_sub, text="フィラー除去")
        self.chk_remove_filler.pack(anchor="w", padx=20, pady=5)
        self.chk_remove_filler.select()

        self.tab_sub = ctk.CTkTabview(self.frame_sub, height=150)
        self.tab_sub.pack(fill="x", padx=5, pady=5)
        
        self.tab_simple = self.tab_sub.add("シンプル設定")
        self.tab_alias = self.tab_sub.add("エイリアス読込 (.object)")

        # --- Tab 1: Simple ---
        ctk.CTkLabel(self.tab_simple, text="フォント:").grid(row=0, column=0, padx=5, sticky="e")
        system_fonts = sorted([f for f in font.families() if not f.startswith("@")])
        self.combo_font = ttk.Combobox(self.tab_simple, values=system_fonts, width=30, state="readonly")
        
        def_font = "TkDefaultFont"
        candidates = ["Yu Gothic UI", "Meiryo", "MS UI Gothic", "Arial", "Segoe UI"]
        for c in candidates:
            if c in system_fonts:
                def_font = c
                break
        self.combo_font.set(def_font)
        self.combo_font.grid(row=0, column=1, columnspan=3, padx=5, sticky="w", pady=5)

        ctk.CTkLabel(self.tab_simple, text="サイズ:").grid(row=1, column=0, padx=5, sticky="e")
        self.entry_size = ctk.CTkEntry(self.tab_simple, width=60)
        self.entry_size.insert(0, "40")
        self.entry_size.grid(row=1, column=1, padx=5, sticky="w")

        ctk.CTkLabel(self.tab_simple, text="文字色:").grid(row=2, column=0, padx=5, sticky="e")
        self.entry_text_color = ctk.CTkEntry(self.tab_simple, width=80)
        self.entry_text_color.insert(0, "000000")
        self.entry_text_color.grid(row=2, column=1, padx=5, sticky="w")
        self.entry_text_color.bind("<KeyRelease>", lambda event: self.update_color_button_from_entry(self.entry_text_color, self.btn_text_color_picker))
        
        self.btn_text_color_picker = ctk.CTkButton(self.tab_simple, text="", width=30, fg_color="#000000", command=self.pick_text_color)
        self.btn_text_color_picker.grid(row=2, column=2, padx=5, sticky="w")

        ctk.CTkLabel(self.tab_simple, text="縁取り色:").grid(row=3, column=0, padx=5, sticky="e")
        self.entry_outline_color = ctk.CTkEntry(self.tab_simple, width=80)
        self.entry_outline_color.insert(0, "ffffff")
        self.entry_outline_color.grid(row=3, column=1, padx=5, sticky="w")
        self.entry_outline_color.bind("<KeyRelease>", lambda event: self.update_color_button_from_entry(self.entry_outline_color, self.btn_outline_color_picker))
        
        self.btn_outline_color_picker = ctk.CTkButton(self.tab_simple, text="", width=30, fg_color="#ffffff", text_color="black", command=self.pick_outline_color)
        self.btn_outline_color_picker.grid(row=3, column=2, padx=5, sticky="w")

        # --- Tab 2: Alias ---
        ctk.CTkLabel(self.tab_alias, text="AviUtlエイリアスファイル (.object):").pack(anchor="w", padx=5, pady=5)
        frm_alias_inner = ctk.CTkFrame(self.tab_alias, fg_color="transparent")
        frm_alias_inner.pack(fill="x", padx=5)
        self.entry_alias = ctk.CTkEntry(frm_alias_inner)
        self.entry_alias.pack(side="left", fill="x", expand=True, padx=(0,5))
        self.make_dnd(self.entry_alias)
        ctk.CTkButton(frm_alias_inner, text="参照", width=60, command=lambda: self.browse_file_generic(self.entry_alias, [("AviUtl Object", "*.object")])).pack(side="right")
        ctk.CTkLabel(self.tab_alias, text="※「テキスト=」の内容だけ差し替えます", font=("", 11), text_color="gray").pack(anchor="w", padx=10)

        # 4. AI Model Settings
        self.frame_model = ctk.CTkFrame(self.frame_left)
        self.frame_model.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(self.frame_model, text="音声認識モデル (Whisper):").pack(anchor="w", padx=5)
        self.combo_whisper = ctk.CTkComboBox(self.frame_model, values=WHISPER_MODELS)
        self.combo_whisper.set("medium")
        self.combo_whisper.pack(fill="x", padx=5, pady=2)

        # 5. 出力ファイル指定
        self.frame_out = ctk.CTkFrame(self.frame_left)
        self.frame_out.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(self.frame_out, text="出力ファイル (任意):").pack(anchor="w", padx=5)
        frm_out_inner = ctk.CTkFrame(self.frame_out, fg_color="transparent")
        frm_out_inner.pack(fill="x", padx=5, pady=2)
        self.entry_output = ctk.CTkEntry(frm_out_inner, placeholder_text="指定なしなら自動命名")
        self.entry_output.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_output = ctk.CTkButton(frm_out_inner, text="保存先を指定", width=100, command=self.browse_save)
        self.btn_output.pack(side="right")

        # 6. 実行ボタン
        self.btn_run = ctk.CTkButton(self.frame_left, text="処理開始 (ACUTE Scripter)", height=60, font=("", 18, "bold"), command=self.run_process)
        self.btn_run.pack(fill="x", pady=20)


        # --- 右カラム ---
        self.frame_right = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_rowconfigure(1, weight=1)

        # 1. Gemini設定
        self.frame_gemini = ctk.CTkFrame(self.frame_right)
        self.frame_gemini.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ctk.CTkLabel(self.frame_gemini, text="■ Gemini AI 補正 (誤字修正・自然分割)", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)
        
        ctk.CTkLabel(self.frame_gemini, text="API Key (入力で有効化):").pack(anchor="w", padx=5)
        self.entry_gemini_key = ctk.CTkEntry(self.frame_gemini, show="*")
        self.entry_gemini_key.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(self.frame_gemini, text="使用モデル (直接入力可):").pack(anchor="w", padx=5)
        self.combo_gemini_model = ctk.CTkComboBox(self.frame_gemini, values=GEMINI_PRESETS)
        self.combo_gemini_model.set("gemini-flash-lite-latest")
        self.combo_gemini_model.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(self.frame_gemini, text="動画の内容 (AIへのヒント):").pack(anchor="w", padx=5)
        self.entry_context = ctk.CTkTextbox(self.frame_gemini, height=100, text_color="gray")
        self.entry_context.pack(fill="x", padx=10, pady=2)
        self.entry_context.insert("1.0", self.ph_context)
        self.entry_context.bind("<FocusIn>", lambda e: self.on_focus_in(self.entry_context, self.ph_context))
        self.entry_context.bind("<FocusOut>", lambda e: self.on_focus_out(self.entry_context, self.ph_context))

        ctk.CTkLabel(self.frame_gemini, text="固有名詞/キーワード (カンマ区切り):").pack(anchor="w", padx=5)
        self.entry_keywords = ctk.CTkTextbox(self.frame_gemini, height=100, text_color="gray")
        self.entry_keywords.pack(fill="x", padx=10, pady=2)
        self.entry_keywords.insert("1.0", self.ph_keywords)
        self.entry_keywords.bind("<FocusIn>", lambda e: self.on_focus_in(self.entry_keywords, self.ph_keywords))
        self.entry_keywords.bind("<FocusOut>", lambda e: self.on_focus_out(self.entry_keywords, self.ph_keywords))

        self.chk_debug = ctk.CTkCheckBox(self.frame_gemini, text="デバッグモード (ログ出力)")
        self.chk_debug.pack(anchor="w", padx=10, pady=10)

        # 2. ログ表示
        self.textbox_log = ctk.CTkTextbox(self.frame_right)
        self.textbox_log.grid(row=1, column=0, sticky="nsew")

        # GUI生成後にジェネレータを初期化
        self.generator = AviUtlGenerator(log_callback=self.log)
        self.on_mode_change() # 初期状態のGUI設定を反映

    def update_color_button_from_entry(self, entry, btn):
        color_code = entry.get().strip().lstrip('#')
        if re.match(r'^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color_code):
            try:
                btn.configure(fg_color=f"#{color_code}")
            except:
                pass

    def pick_text_color(self):
        current_hex = self.entry_text_color.get().strip().lstrip('#')
        try:
            initial = f"#{current_hex}"
        except:
            initial = "#000000"
        color = colorchooser.askcolor(color=initial, title="文字色を選択")
        if color[1]:
            hex_val = color[1].lstrip('#')
            self.entry_text_color.delete(0, "end")
            self.entry_text_color.insert(0, hex_val)
            self.btn_text_color_picker.configure(fg_color=color[1])

    def pick_outline_color(self):
        current_hex = self.entry_outline_color.get().strip().lstrip('#')
        try:
            initial = f"#{current_hex}"
        except:
            initial = "#ffffff"
        color = colorchooser.askcolor(color=initial, title="縁取り色を選択")
        if color[1]:
            hex_val = color[1].lstrip('#')
            self.entry_outline_color.delete(0, "end")
            self.entry_outline_color.insert(0, hex_val)
            self.btn_outline_color_picker.configure(fg_color=color[1])

    def on_focus_in(self, widget, placeholder):
        text = widget.get("1.0", "end-1c")
        if text == placeholder:
            widget.delete("1.0", "end")
            widget.configure(text_color=("black", "white"))

    def on_focus_out(self, widget, placeholder):
        text = widget.get("1.0", "end-1c").strip()
        if not text:
            widget.insert("1.0", placeholder)
            widget.configure(text_color="gray")

    def make_dnd(self, entry_widget):
        entry_widget._entry.drop_target_register(DND_FILES)
        entry_widget._entry.dnd_bind('<<Drop>>', lambda e: self.on_drop(e, entry_widget))

    def browse(self, entry_widget):
        f = filedialog.askopenfilename()
        if f:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, f)

    def browse_file_generic(self, entry_widget, filetypes):
        f = filedialog.askopenfilename(filetypes=filetypes)
        if f:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, f)

    def browse_save(self):
        mode = self.mode_var.get()
        ext = ".srt" if mode == "srt" else ".aup2"
        f = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[(f"{ext.upper()} File", f"*{ext}")])
        if f:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, f)

    def on_drop(self, event, entry_widget):
        path = event.data
        if path.startswith("{") and path.endswith("}"): path = path[1:-1]
        entry_widget.delete(0, "end")
        entry_widget.insert(0, path)

    def on_mode_change(self):
        # SRT出力モードの時は、AUP2設定に関わるタブ内の項目を無効化する
        state = "disabled" if self.mode_var.get() == "srt" else "normal"
        self.combo_font.configure(state=state)
        self.entry_size.configure(state=state)
        self.entry_text_color.configure(state=state)
        self.btn_text_color_picker.configure(state=state)
        self.entry_outline_color.configure(state=state)
        self.btn_outline_color_picker.configure(state=state)
        self.entry_alias.configure(state=state)
        
        # 出力ファイルの拡張子プレースホルダーも変更
        ext = "srt" if self.mode_var.get() == "srt" else "aup2"
        self.entry_output.configure(placeholder_text=f"指定なしなら [動画名]_scripter.{ext}")

    def log(self, text):
        self.textbox_log.insert("end", text + "\n")
        self.textbox_log.see("end")
        self.update()

    def run_process(self):
        try:
            base_file = self.entry_base.get().strip().replace('"', '').replace("'", "")
            if not os.path.exists(base_file):
                messagebox.showerror("エラー", "動画/音声ファイルが見つかりません。")
                return

            mode = self.mode_var.get()
            remove_filler = bool(self.chk_remove_filler.get())
            whisper_size = self.combo_whisper.get()
            gemini_key = self.entry_gemini_key.get().strip()
            gemini_model = self.combo_gemini_model.get().strip()
            gemini_context = self.entry_context.get("1.0", "end-1c").strip()
            if gemini_context == self.ph_context: gemini_context = ""
            gemini_keywords = self.entry_keywords.get("1.0", "end-1c").strip()
            if gemini_keywords == self.ph_keywords: gemini_keywords = ""
            use_gemini = True if gemini_key else False
            is_debug = bool(self.chk_debug.get())

            output_path = self.entry_output.get().strip()
            if not output_path:
                ext = ".srt" if mode == "srt" else ".aup2"
                output_path = os.path.splitext(base_file)[0] + "_scripter" + ext

            self.log(f"=== 処理開始 (ACUTE Scripter: {mode}モード) ===")
            self.log(f"Whisper Model: {whisper_size}")
            
            subtitles = self.generator.transcribe_segment_based(
                file_path=base_file, base_file_path=base_file, model_size=whisper_size, 
                remove_filler=remove_filler, use_gemini=use_gemini, gemini_key=gemini_key, 
                gemini_context=gemini_context, gemini_keywords=gemini_keywords,
                is_debug=is_debug, target_gemini_model=gemini_model
            )

            # --- SRT出力モード ---
            if mode == "srt":
                self.log(f"SRTファイル書き出し中... ({output_path})")
                with open(output_path, "w", encoding="utf-8") as f:
                    for i, sub in enumerate(subtitles):
                        start_str = self.generator.format_srt_time(sub['start'])
                        end_str = self.generator.format_srt_time(sub['end'])
                        f.write(f"{i+1}\n{start_str} --> {end_str}\n{sub['text']}\n\n")
            
            # --- AUP2出力モード ---
            else:
                self.log("AUPデータ計算中...")
                fps = 30.0
                sub_data_list = []
                GAP_LIMIT_SEC = 0.8
                
                for sub in subtitles:
                    s_f = int(sub['start'] * fps)
                    e_f = int(sub['end'] * fps) - 1
                    if e_f < s_f: e_f = s_f
                    sub_data_list.append({"start_f": s_f, "end_f": e_f, "text": sub['text']})
                
                if sub_data_list and sub_data_list[0]['start_f'] < fps * 1.0:
                    sub_data_list[0]['start_f'] = 0

                gap_limit_frames = int(GAP_LIMIT_SEC * fps)
                for i in range(len(sub_data_list) - 1):
                    current_sub = sub_data_list[i]
                    next_sub = sub_data_list[i+1]
                    gap = next_sub['start_f'] - current_sub['end_f'] - 1
                    if 0 < gap < gap_limit_frames:
                        current_sub['end_f'] = next_sub['start_f'] - 1

                self.log(f"AUPファイル書き出し中... ({output_path})")
                body = []
                obj_idx = 0
                
                # Layer 1: ベース動画
                base_duration = self.generator.get_duration(base_file)
                base_end_f = self.generator.sec_to_frame(base_duration, fps) - 1
                if base_end_f <= 0: base_end_f = (sub_data_list[-1]['end_f'] + int(fps*5)) if sub_data_list else 300
                
                body.append(self.generator.create_base_object(obj_idx, layer=1, start_f=0, end_f=base_end_f, file_path=base_file))
                obj_idx += 1

                # Layer 2: 字幕
                current_tab = self.tab_sub.get()
                alias_path = ""
                if "エイリアス" in current_tab:
                    alias_path = self.entry_alias.get().strip().replace('"', '').replace("'", "")
                    if not alias_path or not os.path.exists(alias_path):
                        messagebox.showwarning("注意", "エイリアスファイルが見つからないため、シンプル設定を使用します。")
                        alias_path = ""

                alias_blocks = None
                if alias_path:
                    try:
                        alias_blocks = self.generator.parse_alias(alias_path)
                        self.log(f"エイリアス適用: {os.path.basename(alias_path)}")
                    except Exception as e:
                        self.log(f"エイリアス読込エラー: {e} -> シンプル設定を使用")
                        alias_blocks = None

                if alias_blocks:
                    for sub in sub_data_list:
                        body.append(self.generator.create_subtitle_object_alias(
                            obj_idx, sub['start_f'], sub['end_f'], sub['text'], alias_blocks, layer_id=2
                        ))
                        obj_idx += 1
                else:
                    sub_style = {
                        "font": self.combo_font.get(),
                        "size": float(self.entry_size.get()),
                        "color": self.entry_text_color.get().strip().lstrip('#'),
                        "outline_color": self.entry_outline_color.get().strip().lstrip('#')
                    }
                    for sub in sub_data_list:
                        body.append(self.generator.create_subtitle_object(
                            obj_idx, sub['start_f'], sub['end_f'], sub['text'], sub_style, layer_id=2
                        ))
                        obj_idx += 1

                header = HEADER_TEMPLATE.format(filename="generated.aup2", fps=fps)
                header += "[layer.0]\nlayer=1\nname=Media\n[layer.1]\nlayer=2\nname=Subtitle\n"

                full_content = header + "\n".join(body)
                content_bytes = full_content.encode('utf-8')
                
                with open(output_path, "wb") as f:
                    f.write(content_bytes)

            self.log(f"完了！ファイルを出力しました:\n{output_path}")
            messagebox.showinfo("完了", "処理が完了しました！")

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.log("エラー発生:\n" + err_msg)
            messagebox.showerror("エラー", "処理中にエラーが発生しました。\nログを確認してください。")

if __name__ == "__main__":
    app = AppScripter()
    app.mainloop()