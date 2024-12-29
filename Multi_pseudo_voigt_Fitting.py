from lmfit import Minimizer, Parameters, report_fit, Model
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import csv

__version__ = '1.1.0'

class FittingTool:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Multi Gaussian Fitting    ver: {__version__}")
        
        # UI要素の初期化
        self.init_ui()

    def init_ui(self):
        self.columnshift = 5
        self.rowshift = 11
        
        # グラフ表示用キャンバス
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=1, rowspan=self.rowshift-1, columnspan=self.columnshift-1, padx=10, pady=10)
        
        # ツールバーの作成と表示
        toolbar_frame = tk.Frame(self.root)  # ツールバー用のフレームを作成
        toolbar_frame.grid(row=12, column=0, columnspan=self.columnshift, padx=10, pady=5)  # ツールバーの配置
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # 軸領域用エントリーボックス
        self.range_entries = []
        ttk.Label(self.root, text="Ymax").grid(row=1, column=0, padx=5, pady=5)
        range_entry = ttk.Entry(self.root, width=10, state="normal")
        range_entry.grid(row=2, column=0, padx=5, pady=5)
        self.range_entries.append(range_entry) 
        ttk.Label(self.root, text="Ymin").grid(row=self.rowshift-2, column=0, padx=5, pady=5)
        range_entry = ttk.Entry(self.root, width=10, state="normal")
        range_entry.grid(row=self.rowshift-1, column=0, padx=5, pady=5)
        self.range_entries.append(range_entry) 
        ttk.Label(self.root, text="Xmin").grid(row=self.rowshift, column=1, padx=5, pady=5)
        range_entry = ttk.Entry(self.root, width=10, state="normal")
        range_entry.grid(row=self.rowshift, column=2, padx=5, pady=5)
        self.range_entries.append(range_entry) 
        ttk.Label(self.root, text="Xmax").grid(row=self.rowshift, column=self.columnshift-2, padx=5, pady=5)
        range_entry = ttk.Entry(self.root, width=10, state="normal")
        range_entry.grid(row=self.rowshift, column=self.columnshift-1, padx=5, pady=5)
        self.range_entries.append(range_entry) 
        
        # ファイル選択ボタン
        self.file_button = ttk.Button(self.root, text="Load CSV", command=self.load_csv)
        self.file_button.grid(row=0, column=1, padx=10, pady=10)

        # フィットボタン
        self.fit_button = ttk.Button(self.root, text="Fit", command=self.fit_data)
        self.fit_button.grid(row=0, column=2, padx=10, pady=10)

        # 保存ボタン
        self.save_button = ttk.Button(self.root, text="Save CSV", command=self.save_fitting_results)
        self.save_button.grid(row=0, column=3, padx=10, pady=10)

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
            print("エントリーボックスに正しい数値を入力してください")

    def setup_axis_update(self):
        """エントリーボックスの値が変更された際にグラフを更新"""
        for entry in self.range_entries:
            entry.bind("<FocusOut>", lambda event: self.update_axis_range())
            entry.bind("<Return>", lambda event: self.update_axis_range())

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
            self.ax.errorbar(self.x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars", color='blue')
            self.ax.legend()
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
            checkbox.grid(row=3 + i, column=self.columnshift+1, padx=5, pady=5)
            self.checkboxes.append(check_var)
            
        # χ^2を表示する
        self.X2_entry = []
        ttk.Label(self.root, text="χ^2").grid(row=0, column=self.columnshift+1, padx=5, pady=5)
        X2_entry = ttk.Entry(self.root, width=10, state="readonly")
        X2_entry.grid(row=1, column=self.columnshift+1, padx=5, pady=5)
        self.X2_entry.append(X2_entry) 
        
        # エントリボックスをリストに追加
        for i, label in enumerate(self.bg_labels):
            ttk.Label(self.root, text=label).grid(row=0, column=self.columnshift+2+i, padx=5, pady=5)
            bg_entry = ttk.Entry(self.root, width=10)
            bg_entry.grid(row=1, column=self.columnshift+2+i, padx=5, pady=5)
            bg_entry.insert(0,0)
            self.bg_entries.append(bg_entry)  
            
        # ガウシアンのパラメータ
        for i in range(10):  # 最大10個のガウシアン
            row_entries = []
            # 各ガウシアンのエントリボックス (Area, Center, FWHM)
            for j in range(3):
                entry = ttk.Entry(self.root, width=10)
                entry.grid(row=3 + i, column=self.columnshift+2+j, padx=5, pady=5)
                row_entries.append(entry)
            self.entries.append(row_entries)
            
        # 誤差表示用エントリボックスをリストに追加
        for i, label in enumerate(self.bg_err_labels):
            ttk.Label(self.root, text=label).grid(row=0, column=self.columnshift+5+i, padx=5, pady=5)
            bg_error_entry = ttk.Entry(self.root, width=10, state="readonly")
            bg_error_entry.grid(row=1, column=self.columnshift+5+i, padx=5, pady=5)
            self.bg_errors.append(bg_error_entry)  

        # ガウシアンのパラメータの誤差        
        for i in range(10):  # 最大10個のガウシアン
            row_errors = []
            # 各ガウシアンの誤差表示用エントリボックス (readonly)
            for j in range(3):
                error_entry = ttk.Entry(self.root, width=10, state="readonly")
                error_entry.grid(row=3 + i, column=self.columnshift+5+j, padx=5, pady=5)
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
        ttk.Label(self.root, text="Area").grid(row=2, column=self.columnshift+2)
        ttk.Label(self.root, text="Center").grid(row=2, column=self.columnshift+3)
        ttk.Label(self.root, text="FWHM").grid(row=2, column=self.columnshift+4)
        ttk.Label(self.root, text="Error (Area)").grid(row=2, column=self.columnshift+5)
        ttk.Label(self.root, text="Error (Center)").grid(row=2, column=self.columnshift+6)
        ttk.Label(self.root, text="Error (FWHM)").grid(row=2, column=self.columnshift+7)
    
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
                amp_value, amp_fixed = self.process_param(area)
                wid_value, wid_fixed = self.process_param(fwhm)

                # 'c'がついている場合、固定する処理を追加
                peak_params[f'cen_{i+1}'] = (center_value, center_fixed)
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
            if f'cen_{i+1}' in result.params:
                amp = result.params[f'amp_{i+1}'].value
                cen = result.params[f'cen_{i+1}'].value
                wid = result.params[f'wid_{i+1}'].value

                # ガウス曲線を計算
                peak_y = amp * np.exp(-4 * np.log(2) * ((fit_x_data - cen) / wid)**2) / (wid * (np.pi/(4 * np.log(2)))**(1/2))
                peak_yandBG = bg_a + bg_b * fit_x_data + bg_c * fit_x_data**2 + peak_y

                # 個別のピーク関数を破線でプロット
                self.ax.plot(fit_x_data, peak_yandBG, 'b--', label=f"Gaussian {i+1} fit", color='black')

                # フィット曲線に加算
                y_fit += peak_y

        # グラフを更新
        self.ax.errorbar(x_data, self.y_data, yerr=self.y_error, fmt='o', label="Data with error bars", color='blue')
        self.ax.plot(fit_x_data, y_fit, label="Fitted curve", color='red')
        self.ax.legend()
        
        # 軸範囲を再設定
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
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
                amp = result.params[f'amp_{i+1}'].value
                cen = result.params[f'cen_{i+1}'].value
                wid = result.params[f'wid_{i+1}'].value
                
                # cが付いている場合、末尾に 'c' を追加して表示
                amp_value, amp_fixed = peak_params[f'amp_{i+1}']
                cen_value, cen_fixed = peak_params[f'cen_{i+1}']
                wid_value, wid_fixed = peak_params[f'wid_{i+1}']

                # 'c'が付いている場合、末尾に 'c' を追加して表示
                amp_str = f"{amp:.4f}" + ('c' if amp_fixed else '')
                cen_str = f"{cen:.4f}" + ('c' if cen_fixed else '')
                wid_str = f"{wid:.4f}" + ('c' if wid_fixed else '')

                # 結果をエントリに設定
                for entry in self.entries[i]:
                    entry.config(state="normal")  # 一時的に "normal" に変更
                self.entries[i][0].delete(0, tk.END)
                self.entries[i][1].delete(0, tk.END)
                self.entries[i][2].delete(0, tk.END)

                self.entries[i][0].insert(0, amp_str)
                self.entries[i][1].insert(0, cen_str)
                self.entries[i][2].insert(0, wid_str)

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
        """
        フィッティング結果とフィッティング曲線をCSVファイルに保存する。
        """
        try:
            # フィッティング結果が存在するか確認
            if not hasattr(self, 'result'):
                raise AttributeError("フィッティング結果が存在しません。まずフィットを実行してください。")

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
            gaussian_curves = self.calculate_gaussian_curves(x_fit, fit_params)

            # 保存ダイアログ
            filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")])
            if not filename:
                return  # ファイル名が指定されなかった場合、処理を中断

            with open(filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # フィッティングパラメータを書き込み
                writer.writerow(['Parameter', 'Value', 'Error'])
                for param_name, param in fit_params.items():
                    writer.writerow([param_name, param.value, param.stderr])

                # 空行を追加
                writer.writerow([])

                # データヘッダを書き込み
                header = ['x_data', 'y_data', 'yerr_data' , 'x_fit' , 'y_fit', 'y_bg']
                header += [f'gaussian_{i+1}' for i in range(len(gaussian_curves))]
                writer.writerow(header)

                # データを行ごとに書き込み
                for i, x in enumerate(x_fit):
                    if i < len(x_data):
                        row = [x_data[i], y_data[i], yerr_data[i], x_fit[i] , y_fit[i], y_bg[i]]
                        row += [gaussian[i] for gaussian in gaussian_curves]
                    elif i >= len(x_data):
                        row = ["", "", "" , x_fit[i] , y_fit[i], y_bg[i]]
                        row += [gaussian[i] for gaussian in gaussian_curves]
                    writer.writerow(row)

            messagebox.showinfo("保存完了", "フィッティング結果と曲線を保存しました。")

        except Exception as e:
            messagebox.showerror("エラー", f"保存中にエラーが発生しました: {e}")

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
            amp_key = f'amp_{i}'
            cen_key = f'cen_{i}'
            wid_key = f'wid_{i}'
            if amp_key in params and cen_key in params and wid_key in params:
                amplitude = params[amp_key].value
                center = params[cen_key].value
                width = params[wid_key].value
                gaussian = amplitude * np.exp(-4 * np.log(2) * ((x - center) / width)**2)/ (width * (np.pi/(4 * np.log(2)))**(1/2))
                gaussian_sum += gaussian

        return background + gaussian_sum

    def calculate_gaussian_curves(self, x_data, params):
        """
        各ガウシアン曲線を計算する。
        """
        gaussians = []
        for i in range(1, 11):  # 最大10個のガウシアンを想定
            amp_key = f'amp_{i}'
            cen_key = f'cen_{i}'
            wid_key = f'wid_{i}'
            if amp_key in params and cen_key in params and wid_key in params:
                amplitude = params[amp_key].value
                center = params[cen_key].value
                width = params[wid_key].value
                gaussian = [
                    amplitude * np.exp(-4 * np.log(2) * ((x - center) / width)**2)/ (width * (np.pi/(4 * np.log(2)))**(1/2)) for x in x_data
                ]
                gaussians.append(gaussian)
        return gaussians

if __name__ == "__main__":
    root = tk.Tk()
    app = FittingTool(root)
    root.mainloop()

# cd C:\DATA_HK\python\fitting_software
# pyinstaller -F --noconsole  Multi_pseudo_voigt_Fitting.py
