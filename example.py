# %% 必須: セル区切り


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import curve_fit

# ガウシアン関数
def gaussian(x, amplitude, center, fwhm):
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2)

# バックグラウンド（定数、1次、2次）
def background(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x ** 2

# ガウシアン + バックグラウンド
def combined_function(x, *params):
    num_peaks = (len(params) - 3) // 3
    bg_params = params[:3]
    peak_params = params[3:]
    bg = background(x, *bg_params)
    peaks = sum(
        gaussian(x, peak_params[i], peak_params[i + 1], peak_params[i + 2])
        for i in range(0, len(peak_params), 3)
    )
    return bg + peaks

# メインアプリ
class FittingToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fitting Tool")

        # UI要素の初期化
        self.init_ui()
        
    def toggle_entry_state(self):
        # チェックボックスがオンかオフかによってエントリの状態を変更
        for i, check_var in enumerate(self.checkboxes):
            for j in range(3):
                if check_var.get():
                    self.entries[i][j].config(state="normal")  # チェックされていればエントリを有効化
                else:
                    self.entries[i][j].config(state="readonly")  # チェックが外れていればエントリを無効化
                    
    def init_ui(self):
        # ファイル選択ボタン
        self.file_button = ttk.Button(self.root, text="Load CSV", command=self.load_csv)
        self.file_button.grid(row=0, column=0, padx=10, pady=10)

        # グラフ表示用キャンバス
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=7, padx=10, pady=10)

        # エントリーボックスの作成
        self.create_entry_widgets()

        # フィットボタン
        self.fit_button = ttk.Button(self.root, text="Fit", command=self.fit_data)
        self.fit_button.grid(row=0, column=1, padx=10, pady=10)

        # チェックボックスの初期状態を設定
        self.toggle_entry_state()

    def create_entry_widgets(self):
        self.entries = []
        self.error_entries = []
        self.checkboxes = []
        self.bg_params = [tk.DoubleVar(value=0) for _ in range(3)]
        self.bg_labels = ["Constant", "Linear", "Quadratic"]

        # 背景フィット用ラベルとエントリ
        for i, label in enumerate(self.bg_labels):
            ttk.Label(self.root, text=label).grid(row=2, column=i, padx=5, pady=5)
            entry = ttk.Entry(self.root, textvariable=self.bg_params[i], width=10)
            entry.grid(row=3, column=i, padx=5, pady=5)

        # エントリーボックスとラベル
        ttk.Label(self.root, text="Area").grid(row=5, column=1)
        ttk.Label(self.root, text="Center").grid(row=5, column=2)
        ttk.Label(self.root, text="FWHM").grid(row=5, column=3)
        ttk.Label(self.root, text="Error (Area)").grid(row=5, column=4)
        ttk.Label(self.root, text="Error (Center)").grid(row=5, column=5)
        ttk.Label(self.root, text="Error (FWHM)").grid(row=5, column=6)
        
        # チェックボックスの作成
        for i in range(10):
            check_var = tk.BooleanVar(value=True)
            checkbox = ttk.Checkbutton(self.root, variable=check_var, command=self.toggle_entry_state)
            checkbox.grid(row=6 + i, column=0, padx=5, pady=5)
            self.checkboxes.append(check_var)
            
        # 初期値入力ボックスの作成
        for i in range(10):

            row_entries = []
            row_errors = []

            for j in range(3):
                entry = ttk.Entry(self.root, width=10)
                entry.grid(row=6 + i, column=1 + j, padx=5, pady=5)
                row_entries.append(entry)

            self.entries.append(row_entries)
        
        # 誤差出力ボックスの作成
        for i in range(10):

            row_entries = []
            row_errors = []

            for j in range(3):
                error_entry = ttk.Entry(self.root, width=10, state="readonly")
                error_entry.grid(row=6 + i, column=4 + j, padx=5, pady=5)
                row_errors.append(error_entry)

            self.error_entries.append(row_errors)
            
    def toggle_entry_state(self):
        for i, check_var in enumerate(self.checkboxes):
            state = "normal" if not check_var.get() else "disabled"
            for entry in self.entries[i]:
                entry.config(state=state)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            self.x_data = data[:, 0]
            self.y_data = data[:, 1]

            self.ax.clear()
            self.ax.plot(self.x_data, self.y_data, label="Data")
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")

    def fit_data(self):
        initial_params = []
        bounds = [[], []]

        for param in self.bg_params:
            initial_params.append(param.get())
            bounds[0].append(-np.inf)
            bounds[1].append(np.inf)

        for i, row_entries in enumerate(self.entries):
            if self.checkboxes[i].get():
                continue

            try:
                params = [float(entry.get()) for entry in row_entries]
                initial_params.extend(params)
                bounds[0].extend([0, -np.inf, 0])  # Area >= 0, Center unrestricted, FWHM > 0
                bounds[1].extend([np.inf, np.inf, np.inf])
            except ValueError:
                messagebox.showerror("Error", f"Invalid initial values in row {i + 1}")
                return

        try:
            popt, pcov = curve_fit(
                combined_function, self.x_data, self.y_data,
                p0=initial_params, bounds=bounds
            )

            self.ax.clear()
            self.ax.plot(self.x_data, self.y_data, label="Data")
            self.ax.plot(self.x_data, combined_function(self.x_data, *popt), label="Fit")
            self.ax.legend()
            self.canvas.draw()

            # 結果をUIに反映
            for i, row_entries in enumerate(self.entries):
                if self.checkboxes[i].get():
                    continue

                base_idx = 3 + i * 3
                for j, entry in enumerate(row_entries):
                    entry.delete(0, tk.END)
                    entry.insert(0, f"{popt[base_idx + j]:.4f}")

                for j, error_entry in enumerate(self.error_entries[i]):
                    error = np.sqrt(np.diag(pcov))[base_idx + j]
                    error_entry.config(state="normal")
                    error_entry.delete(0, tk.END)
                    error_entry.insert(0, f"{error:.4f}")
                    error_entry.config(state="readonly")

        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed: {e}")

    

# アプリ起動
root = tk.Tk()
app = FittingToolApp(root)
root.mainloop()


