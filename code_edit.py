def plot_fitted_curve(self, x_data, result):
        # 現在の軸範囲を取得
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        # fittingのデータを滑らかにする。
        fit_x_data = np.arange(np.min(x_data), np.max(x_data), 10*len(x_data))
        
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