import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import lmfit
import csv

# ガウシアン関数
def gaussian(x, amplitude, center, fwhm):
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2)

# バックグラウンド（定数、1次、2次）
def background(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x ** 2

# ガウシアン + バックグラウンド
def combined_function(x, params):
    bg_params = [params['bg_param_0'], params['bg_param_1'], params['bg_param_2']]
    peaks = 0
    num_peaks = (len(params) - 3) // 3  # ガウシアンの数
    for i in range(num_peaks):
        amplitude = params[f'peak_param_{i*3}']
        center = params[f'peak_param_{i*3 + 1}']
        fwhm = params[f'peak_param_{i*3 + 2}']
        peaks += gaussian(x, amplitude, center, fwhm)
    bg = background(x, *bg_params)
    return bg + peaks

class FittingToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fitting Tool")

        self.init_ui()

    def toggle_entry_state(self):
        for i, check_var in enumerate(self.checkboxes):
            state = "normal" if check_var.get() else "readonly"
            for entry in self.entries[i]:
                entry.config(state=state)

    def init_ui(self):
        self.file_button = ttk.Button(self.root, text="Load CSV", command=self.load_csv)
        self.file_button.grid(row=0, column=0, padx=10, pady=10)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=7, padx=10, pady=10)

        self.create_entry_widgets()

        self.fit_button = ttk.Button(self.root, text="Fit", command=self.fit_data)
        self.fit_button.grid(row=0, column=1, padx=10, pady=10)

        self.save_button = ttk.Button(self.root, text="Save CSV", command=self.save_fitting_results)
        self.save_button.grid(row=0, column=2, padx=10, pady=10)

    def create_entry_widgets(self):
        self.entries = []
        self.checkboxes = []
        self.bg_params = [tk.DoubleVar(value=0) for _ in range(3)]
        self.bg_labels = ["Constant", "Linear", "Quadratic"]

        # バックグラウンドフィット用ラベルとエントリ
        for i, label in enumerate(self.bg_labels):
            ttk.Label(self.root, text=label).grid(row=2, column=1+i, padx=5, pady=5)
            bg_entry = ttk.Entry(self.root, textvariable=self.bg_params[i], width=10)
            bg_entry.grid(row=3, column=1+i, padx=5, pady=5)

        # チェックボックスの作成
        for i in range(10):
            check_var = tk.BooleanVar(value=True)
            checkbox = ttk.Checkbutton(self.root, variable=check_var, command=self.toggle_entry_state)
            checkbox.grid(row=6 + i, column=0, padx=5, pady=5)
            self.checkboxes.append(check_var)

        # 初期値入力ボックスの作成
        for i in range(10):
            row_entries = []
            for j in range(3):
                entry = ttk.Entry(self.root, width=10)
                entry.grid(row=6 + i, column=1 + j, padx=5, pady=5)
                row_entries.append(entry)
            self.entries.append(row_entries)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            self.x_data = data[:, 0]
            self.y_data = data[:, 1]
            self.y_error = data[:, 2]

            self.ax.clear()
            self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars")
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")

    def fit_data(self):
        model_params = lmfit.Parameters()

        for i, param in enumerate(self.bg_params):
            model_params.add(f'bg_param_{i}', value=param.get())

        # ガウシアンのパラメータ設定
        for i, row_entries in enumerate(self.entries):
            if self.checkboxes[i].get():
                continue

            try:
                for j, entry in enumerate(row_entries):
                    value_str = entry.get()
                    if value_str.endswith("c"):
                        fixed_value = float(value_str[:-1])
                        model_params.add(f'peak_param_{i*3 + j}', value=fixed_value, vary=False)
                    else:
                        initial_value = float(value_str)
                        model_params.add(f'peak_param_{i*3 + j}', value=initial_value)

            except ValueError:
                pass  # 無効な値を無視

        try:
            # lmfitでフィッティング
            fit_result = lmfit.minimize(self.objective_function, model_params, args=(self.x_data, self.y_data, self.y_error))

            # フィッティング結果を保存
            popt = [fit_result.params[param].value for param in fit_result.params]
            self.update_ui_with_results(popt)

            # プロットを更新
            self.ax.clear()
            self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars")
            self.ax.plot(self.x_data, combined_function(self.x_data, fit_result.params), label="Fit")
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed: {e}")

    def objective_function(self, params, x, y, yerr):
        model = combined_function(x, params)
        return (y - model) / yerr

    def update_ui_with_results(self, popt):
        # フィッティング結果をUIに反映
        for i, row_entries in enumerate(self.entries):
            if self.checkboxes[i].get():
                continue

            base_idx = 3 + i * 3
            for j, entry in enumerate(row_entries):
                entry.delete(0, tk.END)
                entry.insert(0, f"{popt[base_idx + j]:.4f}")

    def save_fitting_results(self):
        if not hasattr(self, 'popt'):
            messagebox.showerror("Error", "No fitting results available to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            popt = self.popt
            num_peaks = (len(popt) - 3) // 3
            combined_fit = combined_function(self.x_data, popt)

            # 保存するデータの作成
            headers = ["x_data", "y_data", "y_error", "combined_fit"]
            for i in range(num_peaks):
                headers.append(f"gaussian_{i+1}_fit")

            output_data = [headers]
            for i, x in enumerate(self.x_data):
                row = [x, self.y_data[i], self.y_error[i], combined_fit[i]]
                output_data.append(row)

            # CSVに保存
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(output_data)

            messagebox.showinfo("Success", f"Fitting results successfully saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save fitting results: {e}")

# アプリケーションの起動
root = tk.Tk()
app = FittingToolApp(root)
root.mainloop()
