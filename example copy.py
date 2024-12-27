from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FittingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Fitting Tool")
        
        # UI要素の初期化
        self.init_ui()

    def init_ui(self):
        # ファイル選択ボタン
        self.file_button = ttk.Button(self.root, text="Load CSV", command=self.load_csv)
        self.file_button.grid(row=0, column=0, padx=10, pady=10)

        # グラフ表示用キャンバス
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=7, padx=10, pady=10)

        # フィットボタン
        self.fit_button = ttk.Button(self.root, text="Fit", command=self.fit_data)
        self.fit_button.grid(row=0, column=1, padx=10, pady=10)

        # 保存ボタン
        self.save_button = ttk.Button(self.root, text="Save CSV", command=self.save_fitting_results)
        self.save_button.grid(row=0, column=2, padx=10, pady=10)

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

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            # CSVファイルを読み込む
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            self.x_data = data[:, 0]
            self.y_data = data[:, 1]
            self.y_error = data[:, 2]

            # プロットを更新
            self.ax.clear()
            self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars")
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")

    def create_entry_widgets(self):
        self.entries = []
        self.bg_entries = []
        self.error_entries = []

        # バックグラウンド項 (定数, 1次, 2次)
        self.bg_entries = []  # バックグラウンドのエントリボックス
        self.bg_errors = []   # バックグラウンドの誤差表示用エントリボックス

        for i, label in enumerate(self.bg_labels):
            ttk.Label(self.root, text=label).grid(row=2, column=1+i, padx=5, pady=5)
            bg_entry = ttk.Entry(self.root, width=10)
            bg_entry.grid(row=3, column=1+i, padx=5, pady=5)
            self.bg_entries.append(bg_entry)  # エントリボックスをリストに追加
        for i, label in enumerate(self.bg_err_labels):
            ttk.Label(self.root, text=label).grid(row=2, column=4+i, padx=5, pady=5)
            bg_error_entry = ttk.Entry(self.root, width=10, state="readonly")
            bg_error_entry.grid(row=3, column=4+i, padx=5, pady=5)
            self.bg_errors.append(bg_error_entry)  # 誤差表示用エントリボックスをリストに追加

        # チェックボックスの作成
        for i in range(10):  # 最大10個のガウシアン
            row_entries = []
            row_errors = []

            # チェックボックス (初期状態でオフ)
            check_var = tk.BooleanVar(value=False)
            checkbox = ttk.Checkbutton(self.root, variable=check_var, command=self.toggle_entry_state)
            checkbox.grid(row=6 + i, column=0, padx=5, pady=5)

            # 各ガウシアンのエントリボックス (Area, Center, FWHM)
            for j in range(3):
                entry = ttk.Entry(self.root, width=10)
                entry.grid(row=6 + i, column=1 + j, padx=5, pady=5)
                row_entries.append(entry)

            # 各ガウシアンの誤差表示用エントリボックス (readonly)
            for j in range(3):
                error_entry = ttk.Entry(self.root, width=10, state="readonly")
                error_entry.grid(row=6 + i, column=4 + j, padx=5, pady=5)
                row_errors.append(error_entry)

            self.entries.append(row_entries)
            self.error_entries.append(row_errors)
            self.checkboxes.append(check_var)

        # パラメータのラベル
        ttk.Label(self.root, text="Area").grid(row=5, column=1)
        ttk.Label(self.root, text="Center").grid(row=5, column=2)
        ttk.Label(self.root, text="FWHM").grid(row=5, column=3)
        ttk.Label(self.root, text="Error (Area)").grid(row=5, column=4)
        ttk.Label(self.root, text="Error (Center)").grid(row=5, column=5)
        ttk.Label(self.root, text="Error (FWHM)").grid(row=5, column=6)


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
                amp = params[f'amp_{i+1}']
                cen = params[f'cen_{i+1}']
                wid = params[f'wid_{i+1}']
                model += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
        
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

                # 'c'がついている場合、固定する処理を追加
                if isinstance(center, str) and center.endswith("c"):
                    center_value = float(center[:-1])  # 'c'を取り除いて値を設定
                    peak_params[f'cen_{i+1}'] = (center_value, True)  # 固定値として設定
                else:
                    peak_params[f'cen_{i+1}'] = (float(center), False)

                # 'c'がついている場合、'c'を取り除いて数値として設定
                amp_value, amp_fixed = self.process_param(area)
                wid_value, wid_fixed = self.process_param(fwhm)

                # 'c'がついている場合、固定する処理を追加
                peak_params[f'amp_{i+1}'] = (amp_value, amp_fixed)
                peak_params[f'wid_{i+1}'] = (wid_value, wid_fixed)
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
        result = mini.leastsq()

        # フィット結果をエントリーボックスに表示
        self.display_fit_results(result)
        
        # フィット結果をグラフに表示
        self.plot_fitted_curve(x_data, result)

    def process_param(self, param):
        """パラメータの 'c' を処理する関数"""
        if isinstance(param, str) and param.endswith("c"):
            value = float(param[:-1])  # 'c'を取り除いて値を設定
            return value, True  # 固定値として設定
        else:
            return float(param), False  # 固定しない値として設定

    def plot_fitted_curve(self, x_data, result):
        """ フィッティング結果をプロットに追加 """
        # バックグラウンドのフィット
        bg_a = result.params['bg_a'].value
        bg_b = result.params['bg_b'].value
        bg_c = result.params['bg_c'].value
        y_fit = bg_a + bg_b * x_data + bg_c * x_data**2

        # 各ピークのガウスフィット
        for i in range(10):
            if f'cen_{i+1}' in result.params:
                amp = result.params[f'amp_{i+1}'].value
                cen = result.params[f'cen_{i+1}'].value
                wid = result.params[f'wid_{i+1}'].value

                # ガウス曲線を追加
                y_fit += amp * np.exp(-(x_data - cen)**2 / (2 * (wid / 2.355)**2))

        # グラフを更新
        self.ax.clear()
        self.ax.errorbar(x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars")
        self.ax.plot(x_data, y_fit, label="Fitted curve", color='red')
        self.ax.legend()
        self.canvas.draw()

    def display_fit_results(self, result):
        """ フィット結果をエントリーボックスに表示 """
        
        # バックグラウンドパラメータの結果を表示
        for entry in self.bg_entries:
            entry.config(state="normal")  # 一時的に "normal" に変更
        self.bg_entries[0].delete(0, tk.END)
        self.bg_entries[1].delete(0, tk.END)
        self.bg_entries[2].delete(0, tk.END)
        self.bg_entries[0].insert(0, f"{result.params['bg_a'].value:.4f}")
        self.bg_entries[1].insert(0, f"{result.params['bg_b'].value:.4f}")
        self.bg_entries[2].insert(0, f"{result.params['bg_c'].value:.4f}")

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
                amp = result.params[f'amp_{i+1}'].value
                cen = result.params[f'cen_{i+1}'].value
                wid = result.params[f'wid_{i+1}'].value

                # 結果をエントリに設定
                for entry in self.entries[i]:
                    entry.config(state="normal")  # 一時的に "normal" に変更
                self.entries[i][0].delete(0, tk.END)
                self.entries[i][1].delete(0, tk.END)
                self.entries[i][2].delete(0, tk.END)

                self.entries[i][0].insert(0, f"{amp:.4f}")
                self.entries[i][1].insert(0, f"{cen:.4f}")
                self.entries[i][2].insert(0, f"{wid:.4f}")

                # 誤差の表示（readonlyに設定）
                for error_entry in self.error_entries[i]:
                    error_entry.config(state="normal")  # 一時的に "normal" に変更
                self.error_entries[i][0].delete(0, tk.END)
                self.error_entries[i][1].delete(0, tk.END)
                self.error_entries[i][2].delete(0, tk.END)

                # 誤差をstderrから取得して表示
                self.error_entries[i][0].insert(0, f"{result.params[f'amp_{i+1}'].stderr:.4f}")
                self.error_entries[i][1].insert(0, f"{result.params[f'cen_{i+1}'].stderr:.4f}")
                self.error_entries[i][2].insert(0, f"{result.params[f'wid_{i+1}'].stderr:.4f}")

                # 最後にエントリを "readonly" に戻す（誤差のみに適用）
                for error_entry in self.error_entries[i]:
                    error_entry.config(state="readonly")

        # 最後にバックグラウンドのエントリを "readonly" に戻す（誤差部分はreadonlyにする）
        for entry in self.bg_entries:
            entry.config(state="normal")
        for entry in self.bg_errors:
            entry.config(state="readonly")  # 誤差部分のみ readonly に戻す


            
    def save_fitting_results(self):
        # フィッティング結果をCSVに保存
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = FittingTool(root)
    root.mainloop()
