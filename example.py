# %% 必須: セル区切り


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import curve_fit
import csv

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
        
        # フィットボタン
        #self.save_button = ttk.Button(self.root, text="Save CSV",command=lambda: self.save_fitting_results(self.popt, self.pcov))
        #self.save_button.grid(row=0, column=2, padx=10, pady=10)
        
        self.save_button = ttk.Button(self.root, text="Save CSV", command=self.save_fitting_results)
        self.save_button.grid(row=0, column=2, padx=10, pady=10)


    def create_entry_widgets(self):
        self.entries = []
        self.error_entries = []
        self.checkboxes = []
        self.bg_params = [tk.DoubleVar(value=0) for _ in range(3)]
        self.bg_labels = ["Constant", "Linear", "Quadratic"]

        # バックグラウンドフィット用ラベルとエントリ
        for i, label in enumerate(self.bg_labels):
            ttk.Label(self.root, text=label).grid(row=2, column=1+i, padx=5, pady=5)
            bg_entry = ttk.Entry(self.root, textvariable=self.bg_params[i], width=10)
            bg_entry.grid(row=3, column=1+i, padx=5, pady=5)
            
        self.err_bg_params = [tk.DoubleVar(value=0) for _ in range(3)]
        self.err_bg_labels = ["Error(Constant)", "Error(Linear)", "Error(Quadratic)"]

        # バックグラウンドフィット用ラベルとエントリ
        for i, label in enumerate(self.err_bg_labels):
            ttk.Label(self.root, text=label).grid(row=2, column=4+i, padx=5, pady=5)
            bg_error_entry = ttk.Entry(self.root, textvariable=self.err_bg_params[i], width=10, state="readonly")
            bg_error_entry.grid(row=3, column=4+i, padx=5, pady=5)

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
            # CSVファイルを読み込む (1列目と2列目はx_data, y_data、3列目はy_error)
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            self.x_data = data[:, 0]  # 1列目がxデータ
            self.y_data = data[:, 1]  # 2列目がyデータ
            self.y_error = data[:, 2]  # 3列目がyエラーバーのデータ

            # プロットを更新
            self.ax.clear()
            self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars")
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")

    def fit_data(self):
        initial_params = []
        bounds = [[], []]

        # バックグラウンドパラメータの初期値と境界を設定
        for param in self.bg_params:
            initial_params.append(param.get())
            bounds[0].append(-np.inf)
            bounds[1].append(np.inf)

        # ガウシアンピークの初期値と境界を設定
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
            # フィッティング
            popt, pcov = curve_fit(
                combined_function, self.x_data, self.y_data,
                p0=initial_params, bounds=bounds
            )

            # 結果を self に保存
            self.popt = popt
            self.pcov = pcov

            # プロットを更新
            self.ax.clear()

            # 元データをエラーバー付きでプロット
            self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars")

            # フィッティング結果をプロット
            self.ax.plot(self.x_data, combined_function(self.x_data, *popt), label="Fit")

            # 個別のガウシアン成分をプロット
            num_peaks = (len(popt) - 3) // 3
            bg_params = popt[:3]
            peak_params = popt[3:]

            # バックグラウンド成分を破線でプロット
            background_curve = background(self.x_data, *bg_params)
            self.ax.plot(self.x_data, background_curve, 'r--', label="Background")
            
            # 背景パラメータをUIに反映
            for i, param_var in enumerate(self.bg_params):
                param_var.set(f"{popt[i]:.4f}")  # フィッティング結果をDoubleVarに設定

            # 背景パラメータの誤差をUIに反映
            for i, error_var in enumerate(self.err_bg_params):
                error = np.sqrt(np.diag(pcov))[i]
                error_var.set(f"{error:.4f}")  # 誤差をDoubleVarに設定

            # 各ガウシアンを破線でプロット
            for i in range(num_peaks):
                amplitude, center, fwhm = peak_params[i * 3:(i + 1) * 3]
                gaussian_curve = gaussian(self.x_data, amplitude, center, fwhm)
                self.ax.plot(self.x_data, gaussian_curve, '--', label=f"Gaussian {i + 1}")

            # 凡例と描画更新
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

            return popt, pcov
        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed: {e}")

    def save_fitting_results(self):
        # フィッティング結果が存在するか確認
        if not hasattr(self, 'popt') or not hasattr(self, 'pcov') or self.popt is None or self.pcov is None:
            messagebox.showerror("Error", "No fitting results available to save.")
            return

        # ファイル保存ダイアログを表示
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Fitting Results"
        )
        if not file_path:  # ファイルが選択されなかった場合
            return

        try:
            # 必要なデータを準備
            popt = self.popt
            pcov = self.pcov
            combined_fit = combined_function(self.x_data, *popt)
            background_fit = background(self.x_data, *popt[:3])
            num_peaks = (len(popt) - 3) // 3
            gaussian_fits = [
                gaussian(self.x_data, *popt[3 + i * 3:3 + (i + 1) * 3]) for i in range(num_peaks)
            ]

            # CSVファイルのヘッダーを準備
            headers = ["x_data", "y_data", "y_error", "combined_fit", "background_fit"]
            headers.extend([f"gaussian_{i + 1}_fit" for i in range(num_peaks)])

            # データ行を構築
            output_data = [headers]
            for i, x in enumerate(self.x_data):
                row = [x, self.y_data[i], self.y_error[i], combined_fit[i], background_fit[i]]
                row.extend(gaussian_fit[i] for gaussian_fit in gaussian_fits)
                output_data.append(row)

            # パラメータのデータを構築
            param_headers = ["parameter", "value", "error"]
            param_data = []

            # バックグラウンドパラメータ
            bg_labels = ["Constant", "Linear", "Quadratic"]
            for i, param in enumerate(popt[:3]):
                param_data.append([bg_labels[i], param, np.sqrt(pcov[i, i])])

            # 各ガウシアンのパラメータ
            for i in range(num_peaks):
                labels = [f"Gaussian {i + 1} Amplitude", f"Gaussian {i + 1} Center", f"Gaussian {i + 1} FWHM"]
                for j, param in enumerate(popt[3 + i * 3:3 + (i + 1) * 3]):
                    param_data.append([labels[j], param, np.sqrt(pcov[3 + i * 3 + j, 3 + i * 3 + j])])

            # CSVファイルに書き込み
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # パラメータのセクション
                writer.writerow(["Fitting Parameters"])
                writer.writerow(param_headers)
                writer.writerows(param_data)
                writer.writerow([])  # 空行を追加

                # フィッティング結果のセクション
                writer.writerow(["Fitted Curves"])
                writer.writerows(output_data)

            # 成功メッセージ
            messagebox.showinfo("Success", f"Fitting results successfully saved to {file_path}")

        except Exception as e:
            # エラーハンドリング
            messagebox.showerror("Error", f"Failed to save fitting results: {e}")

    


# アプリ起動
root = tk.Tk()
app = FittingToolApp(root)
root.mainloop()


