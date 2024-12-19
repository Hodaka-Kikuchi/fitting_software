import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

def gaussian(x, amplitude, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def background(x, a, b, c):
    return a + b * x + c * x ** 2

def model(x, *params):
    num_peaks = (len(params) - 3) // 3  # Subtract 3 for background params
    bg = background(x, params[0], params[1], params[2])
    gaussians = sum(gaussian(x, params[3 + i * 3], params[4 + i * 3], params[5 + i * 3]) for i in range(num_peaks))
    return bg + gaussians

def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        x_data.set(data[:, 0])
        y_data.set(data[:, 1])
        y_error.set(data[:, 2])

        ax.clear()
        ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt="o", label="Data", markersize=3, color="black")
        ax.legend()
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")

def perform_fit():
    try:
        x = np.array(x_data.get())
        y = np.array(y_data.get())
        y_err = np.array(y_error.get())

        # Collect initial guesses and bounds
        params_init = []
        bounds_lower = []
        bounds_upper = []

        for i in range(10):
            if not checkboxes[i].get():
                continue

            amp = float(entries[i][0].get())
            cen = float(entries[i][1].get())
            fwhm = float(entries[i][2].get())

            params_init.extend([amp, cen, fwhm])
            bounds_lower.extend([0, min(x), 0])
            bounds_upper.extend([np.inf, max(x), max(x) - min(x)])

        # Add background parameters
        bg_constant = bg_vars[0].get()
        bg_linear = bg_vars[1].get()
        bg_quadratic = bg_vars[2].get()

        params_init = [bg_constant, bg_linear, bg_quadratic] + params_init
        bounds_lower = [-np.inf, -np.inf, -np.inf] + bounds_lower
        bounds_upper = [np.inf, np.inf, np.inf] + bounds_upper

        # Perform fitting
        popt, pcov = curve_fit(model, x, y, sigma=y_err, p0=params_init, bounds=(bounds_lower, bounds_upper))

        # Update entries with fitted parameters and errors
        errors = np.sqrt(np.diag(pcov))
        for i in range(10):
            if not checkboxes[i].get():
                continue

            entries[i][0].delete(0, tk.END)
            entries[i][0].insert(0, f"{popt[3 + i * 3]:.4f}")
            entries[i][1].delete(0, tk.END)
            entries[i][1].insert(0, f"{popt[4 + i * 3]:.4f}")
            entries[i][2].delete(0, tk.END)
            entries[i][2].insert(0, f"{popt[5 + i * 3]:.4f}")

            errors_entries[i][0].delete(0, tk.END)
            errors_entries[i][0].insert(0, f"{errors[3 + i * 3]:.4f}")
            errors_entries[i][1].delete(0, tk.END)
            errors_entries[i][1].insert(0, f"{errors[4 + i * 3]:.4f}")
            errors_entries[i][2].delete(0, tk.END)
            errors_entries[i][2].insert(0, f"{errors[5 + i * 3]:.4f}")

        # Update graph with fitted curve
        ax.clear()
        ax.errorbar(x, y, yerr=y_err, fmt="o", label="Data", markersize=3, color="black")
        ax.plot(x, model(x, *popt), label="Fit", color="red")
        ax.legend()
        canvas.draw()

        # Save results to CSV
        save_results_to_csv(x, y, y_err, popt, errors)

    except Exception as e:
        messagebox.showerror("Error", f"Fit failed: {e}")

def save_results_to_csv(x, y, y_err, popt, errors):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(["Parameter", "Value", "Error"])

            # Write background parameters
            writer.writerow(["Constant", popt[0], errors[0]])
            writer.writerow(["Linear", popt[1], errors[1]])
            writer.writerow(["Quadratic", popt[2], errors[2]])

            # Write Gaussian parameters
            for i in range(10):
                writer.writerow([f"Peak {i + 1} Amplitude", popt[3 + i * 3], errors[3 + i * 3]])
                writer.writerow([f"Peak {i + 1} Center", popt[4 + i * 3], errors[4 + i * 3]])
                writer.writerow([f"Peak {i + 1} FWHM", popt[5 + i * 3], errors[5 + i * 3]])

            messagebox.showinfo("Success", f"Results saved to {file_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save results: {e}")

root = tk.Tk()
root.title("Fitting Tool")

x_data, y_data, y_error = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()

frame_top = tk.Frame(root)
frame_top.pack(side=tk.TOP, fill=tk.X)

tk.Button(frame_top, text="Load Data", command=load_data).pack(side=tk.LEFT)

bg_frame = tk.Frame(frame_top)
bg_frame.pack(side=tk.LEFT, padx=10)

tk.Label(bg_frame, text="Background: ").pack(side=tk.LEFT)
bg_vars = [tk.DoubleVar(value=0) for _ in range(3)]
tk.Entry(bg_frame, textvariable=bg_vars[0], width=8).pack(side=tk.LEFT)
tk.Entry(bg_frame, textvariable=bg_vars[1], width=8).pack(side=tk.LEFT)
tk.Entry(bg_frame, textvariable=bg_vars[2], width=8).pack(side=tk.LEFT)

tk.Button(frame_top, text="Fit", command=perform_fit).pack(side=tk.RIGHT)

frame_canvas = tk.Frame(root)
frame_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(8, 5))
canvas = FigureCanvasTkAgg(fig, master=frame_canvas)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_bottom = tk.Frame(root)
frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)

checkboxes = []
entries = []
errors_entries = []

for i in range(10):
    row_frame = tk.Frame(frame_bottom)
    row_frame.pack(fill=tk.X)

    check_var = tk.BooleanVar(value=True)  # デフォルトでチェック済み
    check = tk.Checkbutton(row_frame, variable=check_var)
    check.pack(side=tk.LEFT)
    checkboxes.append(check_var)

    entry_row = []
    errors_row = []
    for _ in range(3):  # 面積、中心、FWHM用
        entry = tk.Entry(row_frame, width=10)
        entry.pack(side=tk.LEFT, padx=5)
        entry_row.append(entry)

        error = tk.Entry(row_frame, width=10, state='readonly')  # 誤差用
        error.pack(side=tk.LEFT, padx=5)
        errors_row.append(error)

    entries.append(entry_row)
    error_entries.append(errors_row)


# アプリ起動
root = tk.Tk()
app = FittingToolApp(root)
root.mainloop()