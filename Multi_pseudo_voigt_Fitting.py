from lmfit import Minimizer, Parameters, report_fit, Model
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import csv
from itertools import zip_longest
import sys
import os

__version__ = '1.3.1'

class FittingTool:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Multi Gaussian Fitting    ver: {__version__}")
        
        # UI要素の初期化
        self.init_ui()

    def init_ui(self):
        # ロゴを設定
        # 実行時のリソースパスを設定
        def resource_path(relative_path):
            """PyInstallerでパスを解決する関数"""
            if hasattr(sys, '_MEIPASS'):
                return os.path.join(sys._MEIPASS, relative_path)
            return os.path.join(os.path.abspath("."), relative_path)
        # アイコンの設定
        logo_path = resource_path("logo.ico")
        self.root.iconbitmap(logo_path)
        
        self.columnshift = 1+6
        self.rowshift = 13
        
        # グラフ表示用キャンバス
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=1, rowspan=self.rowshift-2, columnspan=self.columnshift-1, sticky="NSEW")
        
        # ツールバーの作成と表示
        toolbar_frame = tk.Frame(self.root)  # ツールバー用のフレームを作成
        toolbar_frame.grid(row=self.rowshift+1, column=1, columnspan=self.columnshift-1, sticky="NSEW")  # ツールバーの配置
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # 軸領域用エントリーボックス
        self.range_entries = []
        ttk.Label(self.root, text="Ymax").grid(row=2, column=0, sticky="NSEW")
        range_entry = ttk.Entry(self.root, state="normal", width=10)
        range_entry.grid(row=3, column=0, sticky="NSEW")
        self.range_entries.append(range_entry) 
        ttk.Label(self.root, text="Ymin").grid(row=self.rowshift-2, column=0, sticky="NSEW")
        range_entry = ttk.Entry(self.root, state="normal", width=10)
        range_entry.grid(row=self.rowshift-1, column=0, sticky="NSEW")
        self.range_entries.append(range_entry) 
        ttk.Label(self.root, text="Xmin").grid(row=self.rowshift, column=1, sticky="NSEW")
        range_entry = ttk.Entry(self.root, state="normal", width=10)
        range_entry.grid(row=self.rowshift, column=2, sticky="NSEW")
        self.range_entries.append(range_entry) 
        ttk.Label(self.root, text="Xmax").grid(row=self.rowshift, column=self.columnshift-2, sticky="NSEW")
        range_entry = ttk.Entry(self.root, state="normal", width=10)
        range_entry.grid(row=self.rowshift, column=self.columnshift-1, sticky="NSEW")
        self.range_entries.append(range_entry) 
        
        # ファイル選択ボタン
        self.file_button = ttk.Button(self.root, text="Load CSV (data view)", command=self.load_csv_data_view)
        self.file_button.grid(row=0, column=2, sticky="NSEW")
        
        self.file_button = ttk.Button(self.root, text="Load CSV", command=self.load_csv)
        self.file_button.grid(row=0, column=1, sticky="NSEW")
        
        # ファイルcolumnエントリーボックス
        self.data_column_entry = []
        data_column_lbl = ['X Column Index :','Y Column Index :','Yerror Column Index :']
        for i in range(3):
            ttk.Label(self.root, text=data_column_lbl[i]).grid(row=1, column=1+2*i, sticky="NSEW")
            data_column_entry = ttk.Entry(self.root, state="normal", width=10)
            data_column_entry.grid(row=1, column=2+2*i, sticky="NSEW")
            data_column_entry.delete(0, tk.END)
            data_column_entry.insert(0, int(i+1))
            self.data_column_entry.append(data_column_entry) 

        # フィットボタン
        self.fit_button = ttk.Button(self.root, text="Fit", command=self.fit_data)
        self.fit_button.grid(row=0, column=3, sticky="NSEW")

        # 保存ボタン
        self.save_button = ttk.Button(self.root, text="Save CSV", command=self.save_fitting_results)
        self.save_button.grid(row=0, column=4, sticky="NSEW")

        # エントリーボックス作成 (フィッティング用のエントリ)
        self.entries = []
        self.error_entries = []
        self.checkboxes = []
        self.bg_params = [tk.DoubleVar(value=0) for _ in range(3)]  # バックグラウンドパラメータ
        self.bg_labels = ["Constant", "Linear", "Quadratic"]
        self.bg_err_labels = ["Error(Constant)", "Error(Linear)", "Error(Quadratic)"]
        self.create_entry_widgets()
        
        # チェックボックスの初期状態を設定
        self.toggle_entry_state()  # ここで最初に呼び出す
        
        # キャンバスのグラフ表示領域
        self.setup_axis_update()
        
        # グリッドの設定（列と行の重みを均等にする）
        for i in range(13):  # 0-13列までの設定
            self.root.columnconfigure(i, weight=1)
        for i in range(15):  # 0-15行までの設定
            self.root.rowconfigure(i, weight=1)

    def update_axis_range(self):
        """エントリーボックスの値に基づいてグラフの表示範囲を更新"""
        try:
            # エントリーボックスから値を取得
            ymax = float(self.range_entries[0].get()) if self.range_entries[0].get() else None
            ymin = float(self.range_entries[1].get()) if self.range_entries[1].get() else None
            xmin = float(self.range_entries[2].get()) if self.range_entries[2].get() else None
            xmax = float(self.range_entries[3].get()) if self.range_entries[3].get() else None

            # 軸範囲を設定
            if xmin is not None and xmax is not None:
                self.ax.set_xlim(xmin, xmax)
            if ymin is not None and ymax is not None:
                self.ax.set_ylim(ymin, ymax)

            # グラフを更新
            self.canvas.draw()

        except ValueError:
            print("Please enter a valid number in the entry box.")

    def setup_axis_update(self):
        """エントリーボックスの値が変更された際にグラフを更新"""
        for entry in self.range_entries:
            entry.bind("<FocusOut>", lambda event: self.update_axis_range())
            entry.bind("<Return>", lambda event: self.update_axis_range())
    
    # エントリーボックスの数値のcolumnをデータビュー無で読み込み
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            # CSVファイルを読み込む
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                view_data = list(reader)

             # ファイル名の表示
            self.file_name = file_path.split('/')[-1]  # フルパスからファイル名だけを抽出
            
            # ヘッダー行とデータ行を分離
            header = view_data[0]
            rows = view_data[1:]
            
            # columnを自動入力
            x_col = int(float(self.data_column_entry[0].get()))-1
            y_col = int(float(self.data_column_entry[1].get()))-1
            err_col = int(float(self.data_column_entry[2].get()))-1
            
            # ヘッダーをグラフの軸として表示する
            self.X_title = header[x_col]
            self.Y_title = header[y_col]
            
            # データの抽出
            self.x_data = np.array([float(row[x_col]) if len(row) > x_col and row[x_col] != '' and row[x_col] != 'nan' else np.nan for row in rows])
            self.y_data = np.array([float(row[y_col]) if len(row) > y_col and row[y_col] != '' and row[y_col] != 'nan' else np.nan for row in rows])
            self.y_error = np.array([float(row[err_col]) if len(row) > err_col and row[err_col] != '' and row[err_col] != 'nan' else np.nan for row in rows])
            # y_error が 1e-10 以下の場合は 1 に置き換え
            self.y_error = np.where(self.y_error <= 1e-10, 1, self.y_error)
            
            # NaNが含まれている行を削除するためのインデックス作成
            valid_indices = ~np.isnan(self.y_data)  # y_data が NaN でない行を True にするマスク

            # x_data, y_data, y_error をフィルタリング
            self.x_data = self.x_data[valid_indices]
            self.y_data = self.y_data[valid_indices]
            self.y_error = self.y_error[valid_indices]
            
            # プロットを更新
            self.ax.clear()
            self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data", color='blue')
            self.ax.legend()
            # タイトルを設定する。
            self.ax.set_title(f"Selected file: {self.file_name}")
            # x 軸のラベルを設定する。
            self.ax.set_xlabel(self.X_title)
            # y 軸のラベルを設定する。
            self.ax.set_ylabel(self.Y_title)
            self.canvas.draw()
            
            # axis rangeを自動入力
            self.range_entries[0].delete(0, tk.END)
            self.range_entries[0].insert(0, f"{np.max(self.y_data):.4f}")
            self.range_entries[1].delete(0, tk.END)
            self.range_entries[1].insert(0, f"{np.min(self.y_data):.4f}")
            self.range_entries[2].delete(0, tk.END)
            self.range_entries[2].insert(0, f"{np.min(self.x_data):.4f}")
            self.range_entries[3].delete(0, tk.END)
            self.range_entries[3].insert(0, f"{np.max(self.x_data):.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
    
    # データビューモード
    def load_csv_data_view(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            # CSVファイルを読み込む
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                view_data = list(reader)

             # ファイル名の表示
            self.file_name = file_path.split('/')[-1]  # フルパスからファイル名だけを抽出
            
            # ヘッダー行とデータ行を分離
            header = view_data[0]
            rows = view_data[1:]

            # 別ウィンドウでデータプレビューと列選択
            column_selector = tk.Toplevel(self.root)
            column_selector.title("Select Columns for Data")

            # 列番号を表示
            for col_index in range(len(header)):
                tk.Label(column_selector, text=str(col_index + 1), font=("Arial", 10, "bold"), bg="lightgray", borderwidth=1, relief="solid").grid(row=0, column=col_index, sticky="nsew", padx=2, pady=2)

            # ヘッダーを表示
            for col_index, col_name in enumerate(header):
                tk.Label(column_selector, text=col_name, font=("Arial", 10, "bold"), borderwidth=1, relief="solid").grid(row=1, column=col_index, sticky="nsew", padx=2, pady=2)
            
            # データプレビューを表示（最大10行）
            max_preview_rows = 10
            for row_index, row in enumerate(rows[:max_preview_rows], start=2):
                for col_index, value in enumerate(row):
                    tk.Label(column_selector, text=value, borderwidth=1, relief="solid").grid(row=row_index, column=col_index, sticky="nsew", padx=2, pady=2)

            # 列選択エントリ
            tk.Label(column_selector, text="X Column Index :").grid(row=max_preview_rows + 2, column=0, columnspan=2, pady=5, sticky="w")
            x_entry = tk.Entry(column_selector)
            x_entry.grid(row=max_preview_rows + 2, column=2, columnspan=2, pady=5, sticky="w")
            
            tk.Label(column_selector, text="Y Column Index :").grid(row=max_preview_rows + 3, column=0, columnspan=2, pady=5, sticky="w")
            y_entry = tk.Entry(column_selector)
            y_entry.grid(row=max_preview_rows + 3, column=2, columnspan=2, pady=5, sticky="w")
            
            tk.Label(column_selector, text="Yerror Column Index :").grid(row=max_preview_rows + 4, column=0, columnspan=2, pady=5, sticky="w")
            err_entry = tk.Entry(column_selector)
            err_entry.grid(row=max_preview_rows + 4, column=2, columnspan=2, pady=5, sticky="w")

            # 適用ボタン
            def apply_selection():
                try:
                    # ユーザーの入力を取得
                    x_col = int(float(x_entry.get())) - 1
                    y_col = int(float(y_entry.get())) - 1
                    err_col = int(float(err_entry.get())) - 1
                    
                    # ヘッダーをグラフの軸として表示する
                    self.X_title = header[x_col]
                    self.Y_title = header[y_col]

                    # 動的に列データを抽出（数値に変換
                    #self.x_data = [float(row[x_col]) if len(row) > x_col and row[x_col] != '' else None for row in rows]
                    #self.y_data = [float(row[y_col]) if len(row) > y_col and row[y_col] != '' else None for row in rows]
                    #self.y_error = [float(row[err_col]) if len(row) > err_col and row[err_col] != '' else None for row in rows]
                    
                    #np.array([float(row[x_col]) if len(row) > x_col and row[x_col] != '' and row[x_col] != 'nan' else None for row in rows])
                    
                    self.x_data = np.array([float(row[x_col]) if len(row) > x_col and row[x_col] != '' and row[x_col] != 'nan' else np.nan for row in rows])
                    self.y_data = np.array([float(row[y_col]) if len(row) > y_col and row[y_col] != '' and row[y_col] != 'nan' else np.nan for row in rows])
                    self.y_error = np.array([float(row[err_col]) if len(row) > err_col and row[err_col] != '' and row[err_col] != 'nan' else np.nan for row in rows])
                    # y_error が 1e-10 以下の場合は 1 に置き換え
                    self.y_error = np.where(self.y_error <= 1e-10, 1, self.y_error)
                    
                    # NaNが含まれている行を削除するためのインデックス作成
                    valid_indices = ~np.isnan(self.y_data)  # y_data が NaN でない行を True にするマスク

                    # x_data, y_data, y_error をフィルタリング
                    self.x_data = self.x_data[valid_indices]
                    self.y_data = self.y_data[valid_indices]
                    self.y_error = self.y_error[valid_indices]

                    # y_data に nan が含まれている行を削除
                    #valid_indices = ~np.isnan(self.y_data)  # y_data が nan でない行を True にするマスクを作成

                    # x_data, y_data, y_error をマスクでフィルタリング
                    #self.x_data = self.x_data[valid_indices]
                    #self.y_data = self.y_data[valid_indices]
                    #self.y_error = self.y_error[valid_indices]

                    # プロットを更新
                    self.ax.clear()
                    self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data", color='blue')
                    self.ax.legend()
                    # タイトルを設定する。
                    self.ax.set_title(f"Selected file: {self.file_name}")
                    # x 軸のラベルを設定する。
                    self.ax.set_xlabel(self.X_title)
                    # y 軸のラベルを設定する。
                    self.ax.set_ylabel(self.Y_title)
                    self.canvas.draw()
                    
                    # axis rangeを自動入力
                    self.range_entries[0].delete(0, tk.END)
                    self.range_entries[0].insert(0, f"{np.max(self.y_data):.4f}")
                    self.range_entries[1].delete(0, tk.END)
                    self.range_entries[1].insert(0, f"{np.min(self.y_data):.4f}")
                    self.range_entries[2].delete(0, tk.END)
                    self.range_entries[2].insert(0, f"{np.min(self.x_data):.4f}")
                    self.range_entries[3].delete(0, tk.END)
                    self.range_entries[3].insert(0, f"{np.max(self.x_data):.4f}")
                    
                    # columnを自動入力
                    self.data_column_entry[0].delete(0, tk.END)
                    self.data_column_entry[0].insert(0, int(x_col+1))
                    self.data_column_entry[1].delete(0, tk.END)
                    self.data_column_entry[1].insert(0, int(y_col+1))
                    self.data_column_entry[2].delete(0, tk.END)
                    self.data_column_entry[2].insert(0, int(err_col+1)) 

                    # 列選択ウィンドウを閉じる
                    column_selector.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid column selection: {e}")

            apply_button = tk.Button(column_selector, text="Apply", command=apply_selection)
            apply_button.grid(row=max_preview_rows + 6, column=0, columnspan=len(header), pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")            

    """
    # データを全部表示する方が良いと思って作成したが思いのほか使い勝手が悪い。
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            # CSVファイルを読み込む
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                view_data = list(reader)

            # ヘッダー行とデータ行を分離
            header = view_data[0]
            rows = view_data[1:]

            # 別ウィンドウでデータプレビューと列選択
            column_selector = tk.Toplevel(self.root)
            column_selector.title("Select Columns for Data")

            # キャンバスとスクロールバーを作成
            canvas = tk.Canvas(column_selector)
            scrollbar = tk.Scrollbar(column_selector, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            # スクロール可能なフレームの設定
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            # キャンバスにフレームを配置
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # スクロールバーの配置
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            # 列番号を表示
            for col_index in range(len(header)):
                tk.Label(scrollable_frame, text=str(col_index + 1), font=("Arial", 10, "bold"), bg="lightgray", borderwidth=1, relief="solid").grid(row=0, column=col_index, sticky="nsew", padx=2, pady=2)

            # ヘッダーを表示
            for col_index, col_name in enumerate(header):
                tk.Label(scrollable_frame, text=col_name, font=("Arial", 10, "bold"), borderwidth=1, relief="solid").grid(row=1, column=col_index, sticky="nsew", padx=2, pady=2)

            # データをスクロール可能な領域に表示
            for row_index, row in enumerate(rows, start=2):
                for col_index, value in enumerate(row):
                    tk.Label(scrollable_frame, text=value, borderwidth=1, relief="solid").grid(row=row_index, column=col_index, sticky="nsew", padx=2, pady=2)

            # 列選択エントリ
            tk.Label(scrollable_frame, text="X Column Index :").grid(row=len(rows) + 2, column=0, columnspan=2, pady=5, sticky="w")
            x_entry = tk.Entry(scrollable_frame)
            x_entry.grid(row=len(rows) + 2, column=2, columnspan=2, pady=5, sticky="w")
            
            tk.Label(scrollable_frame, text="Y Column Index :").grid(row=len(rows) + 3, column=0, columnspan=2, pady=5, sticky="w")
            y_entry = tk.Entry(scrollable_frame)
            y_entry.grid(row=len(rows) + 3, column=2, columnspan=2, pady=5, sticky="w")
            
            tk.Label(scrollable_frame, text="Yerror Column Index :").grid(row=len(rows) + 4, column=0, columnspan=2, pady=5, sticky="w")
            err_entry = tk.Entry(scrollable_frame)
            err_entry.grid(row=len(rows) + 4, column=2, columnspan=2, pady=5, sticky="w")

            # 適用ボタン
            def apply_selection():
                try:
                    # ユーザーの入力を取得
                    x_col = int(float(x_entry.get())) - 1
                    y_col = int(float(y_entry.get())) - 1
                    err_col = int(float(err_entry.get())) - 1

                    # ヘッダーをグラフの軸として表示する
                    self.X_title = header[x_col]
                    self.Y_title = header[y_col]

                    # 動的に列データを抽出（数値に変換）
                    self.x_data = np.array([float(row[x_col]) if len(row) > x_col and row[x_col] != '' and row[x_col] != 'nan' else np.nan for row in rows])
                    self.y_data = np.array([float(row[y_col]) if len(row) > y_col and row[y_col] != '' and row[y_col] != 'nan' else np.nan for row in rows])
                    self.y_error = np.array([float(row[err_col]) if len(row) > err_col and row[err_col] != '' and row[err_col] != 'nan' else np.nan for row in rows])

                    # y_error が 1e-10 以下の場合は 1 に置き換え
                    self.y_error = np.where(self.y_error <= 1e-10, 1, self.y_error)

                    # NaNが含まれている行を削除するためのインデックス作成
                    valid_indices = ~np.isnan(self.y_data)  # y_data が NaN でない行を True にするマスク

                    # x_data, y_data, y_error をフィルタリング
                    self.x_data = self.x_data[valid_indices]
                    self.y_data = self.y_data[valid_indices]
                    self.y_error = self.y_error[valid_indices]

                    # プロットを更新
                    self.ax.clear()
                    self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data", color='blue')
                    self.ax.legend()
                    self.ax.set_xlabel(self.X_title)
                    self.ax.set_ylabel(self.Y_title)
                    self.canvas.draw()

                    # range_entriesを更新
                    self.range_entries[0].delete(0, tk.END)
                    self.range_entries[0].insert(0, f"{np.max(self.y_data):.4f}")
                    self.range_entries[1].delete(0, tk.END)
                    self.range_entries[1].insert(0, f"{np.min(self.y_data):.4f}")
                    self.range_entries[2].delete(0, tk.END)
                    self.range_entries[2].insert(0, f"{np.min(self.x_data):.4f}")
                    self.range_entries[3].delete(0, tk.END)
                    self.range_entries[3].insert(0, f"{np.max(self.x_data):.4f}")

                    # 列選択ウィンドウを閉じる
                    column_selector.destroy()

                except Exception as e:
                    messagebox.showerror("Error", f"Invalid column selection: {e}")

            apply_button = tk.Button(scrollable_frame, text="Apply", command=apply_selection)
            apply_button.grid(row=len(rows) + 6, column=0, columnspan=len(header), pady=10)

            # ウィンドウサイズを自動調整
            column_selector.update_idletasks()
            width = max((len(header) + 1) * 100, 600)  # 列数に基づく幅の設定
            height = min(len(rows) * 25, 600)  # 行数に基づく高さの設定
            column_selector.geometry(f"{width}x{height}")

            # キャンバスでマウススクロールを操作できるようにする
            def on_mouse_wheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            canvas.bind_all("<MouseWheel>", on_mouse_wheel)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            
    """
        
    def create_entry_widgets(self):
        self.entries = []
        self.bg_entries = []
        self.error_entries = []

        # バックグラウンド項 (定数, 1次, 2次)
        self.bg_entries = []  # バックグラウンドのエントリボックス
        self.bg_errors = []   # バックグラウンドの誤差表示用エントリボックス
        
        # チェックボックスの作成
        for i in range(10):  # 最大10個のガウシアン
            # チェックボックス (初期状態でオフ)
            check_var = tk.BooleanVar(value=False)
            checkbox = ttk.Checkbutton(self.root, variable=check_var, command=self.toggle_entry_state)
            checkbox.grid(row=3 + i, column=self.columnshift+1, sticky="NSEW")
            self.checkboxes.append(check_var)
            
        # χ^2を表示する
        self.X2_entry = []
        ttk.Label(self.root, text="χ^2").grid(row=0, column=self.columnshift+1, sticky="NSEW")
        X2_entry = ttk.Entry(self.root, state="readonly", width=10)
        X2_entry.grid(row=1, column=self.columnshift+1, sticky="NSEW")
        self.X2_entry.append(X2_entry) 
        
        # エントリボックスをリストに追加
        for i, label in enumerate(self.bg_labels):
            ttk.Label(self.root, text=label).grid(row=0, column=self.columnshift+2+i, sticky="NSEW")
            bg_entry = ttk.Entry(self.root, width=10)
            bg_entry.grid(row=1, column=self.columnshift+2+i, sticky="NSEW")
            bg_entry.insert(0,0)
            self.bg_entries.append(bg_entry)  
            
        # ガウシアンのパラメータ
        for i in range(10):  # 最大10個のガウシアン
            row_entries = []
            # 各ガウシアンのエントリボックス (Area, Center, FWHM)
            for j in range(3):
                entry = ttk.Entry(self.root, width=10)
                entry.grid(row=3 + i, column=self.columnshift+2+j, sticky="NSEW")
                row_entries.append(entry)
            self.entries.append(row_entries)
            
        # 誤差表示用エントリボックスをリストに追加
        for i, label in enumerate(self.bg_err_labels):
            ttk.Label(self.root, text=label).grid(row=0, column=self.columnshift+5+i, sticky="NSEW")
            bg_error_entry = ttk.Entry(self.root, state="readonly", width=10)
            bg_error_entry.grid(row=1, column=self.columnshift+5+i, sticky="NSEW")
            self.bg_errors.append(bg_error_entry)  

        # ガウシアンのパラメータの誤差        
        for i in range(10):  # 最大10個のガウシアン
            row_errors = []
            # 各ガウシアンの誤差表示用エントリボックス (readonly)
            for j in range(3):
                error_entry = ttk.Entry(self.root, state="readonly", width=10)
                error_entry.grid(row=3 + i, column=self.columnshift+5+j, sticky="NSEW")
                row_errors.append(error_entry)
            self.error_entries.append(row_errors)
            
        # パラメータのラベル
        """
        ttk.Label(self.root, text="G_ratio").grid(row=2, column=4)
        ttk.Label(self.root, text="Area").grid(row=2, column=5)
        ttk.Label(self.root, text="Center").grid(row=2, column=6)
        ttk.Label(self.root, text="G_FWHM").grid(row=2, column=7)
        ttk.Label(self.root, text="L_FWHM").grid(row=2, column=8)
        ttk.Label(self.root, text="Error (G_ratio)").grid(row=2, column=9)
        ttk.Label(self.root, text="Error (Area)").grid(row=2, column=10)
        ttk.Label(self.root, text="Error (Center)").grid(row=2, column=11)
        ttk.Label(self.root, text="Error (G_FWHM)").grid(row=2, column=12)
        ttk.Label(self.root, text="Error (L_FWHM)").grid(row=2, column=13)
        """
        ttk.Label(self.root, text="Area").grid(row=2, column=self.columnshift+2, sticky="NSEW")
        ttk.Label(self.root, text="Center").grid(row=2, column=self.columnshift+3, sticky="NSEW")
        ttk.Label(self.root, text="FWHM").grid(row=2, column=self.columnshift+4, sticky="NSEW")
        ttk.Label(self.root, text="Error (Area)").grid(row=2, column=self.columnshift+5, sticky="NSEW")
        ttk.Label(self.root, text="Error (Center)").grid(row=2, column=self.columnshift+6, sticky="NSEW")
        ttk.Label(self.root, text="Error (FWHM)").grid(row=2, column=self.columnshift+7, sticky="NSEW")
    
    def toggle_entry_state(self):
        """ チェックボックスの状態に応じてエントリの有効化・無効化 """
        for i in range(10):
            state = "normal" if self.checkboxes[i].get() else "readonly"
            for entry in self.entries[i]:
                entry.config(state=state)

    def residual(self, params, x, y, y_err):
        """ フィット関数の残差計算 """
        # バックグラウンド項
        bg_a = params['bg_a']
        bg_b = params['bg_b']
        bg_c = params['bg_c']
        model = bg_a + bg_b * x + bg_c * x**2

        # ガウシアン項
        for i in range(10):
            if self.checkboxes[i].get():
                amp = params[f'area_{i+1}']
                cen = params[f'center_{i+1}']
                wid = params[f'FWHM_{i+1}']
                model += amp * np.exp(-4 * np.log(2) * ((x - cen) / wid)**2)/ (wid * (np.pi/(4 * np.log(2)))**(1/2))
                # amp * np.exp(-4 * np.log(2) * ((x_data - cen) / wid)**2)/ (wid * (np.pi/(4 * np.log(2)))**(1/2))
        
        return (y - model) / y_err  # ノイズを無視するために誤差で割る

    def fit_data(self):
        # バックグラウンドパラメータの取得と処理
        bg_a = self.bg_entries[0].get()
        bg_b = self.bg_entries[1].get()
        bg_c = self.bg_entries[2].get()

        # バックグラウンドパラメータの 'c' を取り除く処理
        bg_a_value, bg_a_fixed = self.process_param(bg_a)
        bg_b_value, bg_b_fixed = self.process_param(bg_b)
        bg_c_value, bg_c_fixed = self.process_param(bg_c)
        
        # 各ピークに対するパラメータの取得とフィッティング
        peak_params = {}
        for i in range(10):  # 最大10個のピークに対して
            if self.checkboxes[i].get():  # チェックボックスがオンの場合のみ
                area = self.entries[i][0].get()
                center = self.entries[i][1].get()
                fwhm = self.entries[i][2].get()

                # 'c'がついている場合、'c'を取り除いて数値として設定
                center_value, center_fixed = self.process_param(center)
                area_value, area_fixed = self.process_param(area)
                FWHM_value, FWHM_fixed = self.process_param(fwhm)

                # 'c'がついている場合、固定する処理を追加
                peak_params[f'center_{i+1}'] = (center_value, center_fixed)
                peak_params[f'area_{i+1}'] = (area_value, area_fixed)
                peak_params[f'FWHM_{i+1}'] = (FWHM_value, FWHM_fixed)
            else:
                continue
        
        # フィットするデータ
        x_data = self.x_data
        y_data = self.y_data
        y_error = self.y_error
        
        # lmfitの最小化処理
        pfit = Parameters()

        # バックグラウンドパラメータを追加（固定値かどうかをチェック）
        pfit.add('bg_a', value=bg_a_value, vary=not bg_a_fixed)
        pfit.add('bg_b', value=bg_b_value, vary=not bg_b_fixed)
        pfit.add('bg_c', value=bg_c_value, vary=not bg_c_fixed)

        # ガウシアンのピークパラメータを追加
        for key, value in peak_params.items():
            pfit.add(key, value=value[0], vary=not value[1])

        # 最小化処理
        mini = Minimizer(self.residual, pfit, fcn_args=(x_data, y_data, y_error))
        self.result = mini.leastsq()

        # フィット結果をエントリーボックスに表示
        self.display_fit_results(self.result, bg_a_fixed, bg_b_fixed, bg_c_fixed,peak_params)
        
        # フィット結果をグラフに表示
        self.plot_fitted_curve(x_data, self.result)

    def process_param(self, param):
        """パラメータの 'c' を処理する関数"""
        if isinstance(param, str) and param.endswith("c"):
            value = float(param[:-1])  # 'c'を取り除いて値を設定
            return value, True  # 固定値として設定
        else:
            return float(param), False  # 固定しない値として設定

    def plot_fitted_curve(self, x_data, result):
        # 現在の軸範囲を取得
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        # fittingのデータを滑らかにする。
        fit_x_data = np.arange(np.min(x_data), np.max(x_data), (np.max(x_data) - np.min(x_data))/(10*len(x_data)))
        self.fit_x_data = fit_x_data
        
        self.ax.clear()
        """ フィッティング結果をプロットに追加 """
        # バックグラウンドのフィット
        bg_a = result.params['bg_a'].value
        bg_b = result.params['bg_b'].value
        bg_c = result.params['bg_c'].value
        y_fit = bg_a + bg_b * fit_x_data + bg_c * fit_x_data**2

        # バックグラウンド関数を破線でプロット
        self.ax.plot(fit_x_data, y_fit, 'r--', label="Background fit", color='yellow')

        # 各ピークのガウスフィット
        for i in range(10):
            if f'center_{i+1}' in result.params:
                amp = result.params[f'area_{i+1}'].value
                cen = result.params[f'center_{i+1}'].value
                wid = result.params[f'FWHM_{i+1}'].value

                # ガウス曲線を計算
                peak_y = amp * np.exp(-4 * np.log(2) * ((fit_x_data - cen) / wid)**2) / (wid * (np.pi/(4 * np.log(2)))**(1/2))
                peak_yandBG = bg_a + bg_b * fit_x_data + bg_c * fit_x_data**2 + peak_y

                # 個別のピーク関数を破線でプロット
                self.ax.plot(fit_x_data, peak_yandBG, 'b--', label=f"Gaussian {i+1} fit", color='black')

                # フィット曲線に加算
                y_fit += peak_y

        # グラフを更新
        self.ax.errorbar(x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data", color='blue')
        self.ax.plot(fit_x_data, y_fit, label="Fitted curve", color='red')
        self.ax.legend()
        
        # 軸範囲を再設定
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        # タイトル
        self.ax.set_title(f"Selected file: {self.file_name}")
        # x 軸のラベルを設定する。
        self.ax.set_xlabel(self.X_title)
        # y 軸のラベルを設定する。
        self.ax.set_ylabel(self.Y_title)
        
        self.canvas.draw()

    def display_fit_results(self, result, bg_a_fixed, bg_b_fixed, bg_c_fixed, peak_params):
        """ フィット結果をエントリーボックスに表示 """
        # χ^2を表示
        self.X2_entry[0].config(state="normal")  # 一時的に "normal" に変更
        self.X2_entry[0].delete(0, tk.END)
        self.X2_entry[0].insert(0, f"{result.redchi:.4f}")
        self.X2_entry[0].config(state="readonly")
        
        # バックグラウンドパラメータの結果を表示
        for entry in self.bg_entries:
            entry.config(state="normal")  # 一時的に "normal" に変更
        self.bg_entries[0].delete(0, tk.END)
        self.bg_entries[1].delete(0, tk.END)
        self.bg_entries[2].delete(0, tk.END)
        self.bg_entries[0].insert(0, f"{result.params['bg_a'].value:.4f}"+ ('c' if bg_a_fixed else ''))
        self.bg_entries[1].insert(0, f"{result.params['bg_b'].value:.4f}"+ ('c' if bg_b_fixed else ''))
        self.bg_entries[2].insert(0, f"{result.params['bg_c'].value:.4f}"+ ('c' if bg_c_fixed else ''))

        # 誤差の表示（readonlyに設定）
        for entry in self.bg_errors:
            entry.config(state="normal")  # 一時的に "normal" に変更
        self.bg_errors[0].delete(0, tk.END)
        self.bg_errors[1].delete(0, tk.END)
        self.bg_errors[2].delete(0, tk.END)
        self.bg_errors[0].insert(0, f"{result.params['bg_a'].stderr:.4f}")
        self.bg_errors[1].insert(0, f"{result.params['bg_b'].stderr:.4f}")
        self.bg_errors[2].insert(0, f"{result.params['bg_c'].stderr:.4f}")

        # ガウシアンパラメータの結果を表示
        for i in range(10):
            if self.checkboxes[i].get():
                # 各ピークの結果をエントリに設定
                amp = result.params[f'area_{i+1}'].value
                cen = result.params[f'center_{i+1}'].value
                wid = result.params[f'FWHM_{i+1}'].value
                
                # cが付いている場合、末尾に 'c' を追加して表示
                area_value, area_fixed = peak_params[f'area_{i+1}']
                center_value, center_fixed = peak_params[f'center_{i+1}']
                FWHM_value, FWHM_fixed = peak_params[f'FWHM_{i+1}']

                # 'c'が付いている場合、末尾に 'c' を追加して表示
                area_str = f"{amp:.4f}" + ('c' if area_fixed else '')
                center_str = f"{cen:.4f}" + ('c' if center_fixed else '')
                FWHM_str = f"{wid:.4f}" + ('c' if FWHM_fixed else '')

                # 結果をエントリに設定
                for entry in self.entries[i]:
                    entry.config(state="normal")  # 一時的に "normal" に変更
                self.entries[i][0].delete(0, tk.END)
                self.entries[i][1].delete(0, tk.END)
                self.entries[i][2].delete(0, tk.END)

                self.entries[i][0].insert(0, area_str)
                self.entries[i][1].insert(0, center_str)
                self.entries[i][2].insert(0, FWHM_str)

                # 誤差の表示（readonlyに設定）
                for error_entry in self.error_entries[i]:
                    error_entry.config(state="normal")  # 一時的に "normal" に変更
                self.error_entries[i][0].delete(0, tk.END)
                self.error_entries[i][1].delete(0, tk.END)
                self.error_entries[i][2].delete(0, tk.END)

                # 誤差をstderrから取得して表示
                self.error_entries[i][0].insert(0, f"{result.params[f'area_{i+1}'].stderr:.4f}")
                self.error_entries[i][1].insert(0, f"{result.params[f'center_{i+1}'].stderr:.4f}")
                self.error_entries[i][2].insert(0, f"{result.params[f'FWHM_{i+1}'].stderr:.4f}")
                
                # 最後にエントリを "readonly" に戻す（誤差のみに適用）
                for error_entry in self.error_entries[i]:
                    error_entry.config(state="readonly")

        # 最後にバックグラウンドのエントリを "readonly" に戻す（誤差部分はreadonlyにする）
        for entry in self.bg_entries:
            entry.config(state="normal")
        for entry in self.bg_errors:
            entry.config(state="readonly")  # 誤差部分のみ readonly に戻す
    
    def save_fitting_results(self):
        """
        フィッティング結果とフィッティング曲線をCSVファイルに保存する。
        """
        try:
            # フィッティング結果が存在するか確認
            if not hasattr(self, 'result'):
                raise AttributeError("Fitting results do not exist. Please perform fitting first.")

            result = self.result
            fit_params = result.params

            # 元データ
            x_data = self.x_data
            y_data = self.y_data
            yerr_data = self.y_error
            x_fit = self.fit_x_data

            # フィッティング曲線の計算
            y_fit = self.calculate_fit_curve(x_fit, fit_params)

            # バックグラウンド曲線の計算
            y_bg = self.calculate_background_curve(x_fit, fit_params)

            # 各ガウシアン曲線の計算
            # gaussian_curves = self.calculate_gaussian_curves(x_fit, fit_params) # BG無
            gaussian_curves = self.calculate_gaussian_and_BG_curves(x_fit, fit_params) # BG有

            # 保存ダイアログ
            filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")])
            if not filename:
                return  # ファイル名が指定されなかった場合、処理を中断

            with open(filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # χ²（カイ二乗）の値を書き込む
                chi2_value = result.redchi  # χ²の値を取得
                # Chi-squaredとパラメータ用のデータを準備
                param_rows = [['Chi-squared', chi2_value, '']]
                #param_rows.append(['Parameter', 'Value', 'Error'])
                for param_name, param in fit_params.items():
                    param_rows.append([param_name, param.value, param.stderr])

                # データ列の準備
                data_headers = ['','x_data', 'y_data', 'yerr_data', 'x_fit', 'y_fit', 'y_bg']  # 空列を追加
                data_headers += [f'gaussian_{i + 1}' for i in range(len(gaussian_curves))]

                # 各データ列を同じ長さにするため調整
                max_length = max(len(x_data), len(x_fit))
                x_data = list(x_data) + [""] * (max_length - len(x_data))
                y_data = list(y_data) + [""] * (max_length - len(y_data))
                yerr_data = list(yerr_data) + [""] * (max_length - len(yerr_data))
                x_fit = list(x_fit) + [""] * (max_length - len(x_fit))
                y_fit = list(y_fit) + [""] * (max_length - len(y_fit))
                y_bg = list(y_bg) + [""] * (max_length - len(y_bg))
                gaussian_curves = [list(gaussian) + [""] * (max_length - len(gaussian)) for gaussian in gaussian_curves]

                # データ列を行ごとにまとめる
                data_rows = list(zip(x_data, y_data, yerr_data, x_fit, y_fit, y_bg, *gaussian_curves))

                # ヘッダー行を作成
                header_row = ['Parameter', 'Value', 'Error'] + data_headers

                # ヘッダー行を書き込み
                writer.writerow(header_row)

                # パラメータ行とデータ行を列方向に統合して書き込み
                for i in range(max(max_length, len(param_rows))):
                    param_part = param_rows[i] if i < len(param_rows) else [""] * 3
                    data_part = list(data_rows[i]) if i < len(data_rows) else [""] * len(data_headers)
                    writer.writerow(param_part + [""] + data_part)  # 空列を追加
                
            messagebox.showinfo("Save Complete", "Fitting results and curves have been saved.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving.: {e}")

    def calculate_fit_curve(self, x_data, params):
        """
        フィッティング曲線を計算する。
        """
        return [self.model(params, x) for x in x_data]

    def calculate_background_curve(self, x_data, params):
        """
        バックグラウンド曲線を計算する。
        """
        bg_a = params['bg_a'].value
        bg_b = params['bg_b'].value
        bg_c = params['bg_c'].value
        return [bg_a + bg_b * x + bg_c * (x**2) for x in x_data]

    def model(self, params, x):
        """
        モデル関数：バックグラウンド + ガウシアンの合計を計算する。
        """
        # バックグラウンド部分
        bg_a = params['bg_a'].value
        bg_b = params['bg_b'].value
        bg_c = params['bg_c'].value
        background = bg_a + bg_b * x + bg_c * (x**2)

        # ガウシアン部分
        gaussian_sum = 0
        for i in range(1, 11):  # 最大10個のガウシアン
            area_key = f'area_{i}'
            center_key = f'center_{i}'
            FWHM_key = f'FWHM_{i}'
            if area_key in params and center_key in params and FWHM_key in params:
                amplitude = params[area_key].value
                center = params[center_key].value
                width = params[FWHM_key].value
                gaussian = amplitude * np.exp(-4 * np.log(2) * ((x - center) / width)**2)/ (width * (np.pi/(4 * np.log(2)))**(1/2))
                gaussian_sum += gaussian

        return background + gaussian_sum

    def calculate_gaussian_curves(self, x_data, params):
        """
        各ガウシアン曲線を計算する。
        """
        gaussians = []
        for i in range(1, 11):  # 最大10個のガウシアンを想定
            area_key = f'area_{i}'
            center_key = f'center_{i}'
            FWHM_key = f'FWHM_{i}'
            if area_key in params and center_key in params and FWHM_key in params:
                amplitude = params[area_key].value
                center = params[center_key].value
                width = params[FWHM_key].value
                gaussian = [
                    amplitude * np.exp(-4 * np.log(2) * ((x - center) / width)**2)/ (width * (np.pi/(4 * np.log(2)))**(1/2)) for x in x_data
                ]
                gaussians.append(gaussian)
        return gaussians
    
    def calculate_gaussian_and_BG_curves(self, x_data, params):
        bg_a = params['bg_a'].value
        bg_b = params['bg_b'].value
        bg_c = params['bg_c'].value
        gaussians = []
        for i in range(1, 11):  # 最大10個のガウシアンを想定
            area_key = f'area_{i}'
            center_key = f'center_{i}'
            FWHM_key = f'FWHM_{i}'
            if area_key in params and center_key in params and FWHM_key in params:
                amplitude = params[area_key].value
                center = params[center_key].value
                width = params[FWHM_key].value
                gaussian = [
                    amplitude * np.exp(-4 * np.log(2) * ((x - center) / width)**2)/ (width * (np.pi/(4 * np.log(2)))**(1/2)) + bg_a + bg_b * x + bg_c * (x**2) for x in x_data
                ]
                gaussians.append(gaussian)
        return gaussians
    
if __name__ == "__main__":
    root = tk.Tk()
    app = FittingTool(root)
    root.mainloop()

# cd C:\DATA_HK\python\fitting_software
# pyinstaller -F --noconsole --add-data "logo.ico;." --icon=logo.ico Multi_pseudo_voigt_Fitting.py