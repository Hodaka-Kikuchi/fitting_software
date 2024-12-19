# %% 必須: セル区切り


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit

def gaussian(x, amplitude, center, fwhm):
    return amplitude * np.exp(-4 * np.log(2) * ((x - center) / fwhm) ** 2)

def fit_data():
    global x_data, y_data
    params = []
    bounds_lower = []
    bounds_upper = []
    
    for i in range(10):
        if not check_vars[i].get():
            try:
                amp = float(amplitude_entries[i].get())
                cen = float(center_entries[i].get())
                fwhm = float(fwhm_entries[i].get())
                params.extend([amp, cen, fwhm])
                bounds_lower.extend([0, min(x_data), 0])
                bounds_upper.extend([np.inf, max(x_data), np.inf])
            except ValueError:
                messagebox.showerror("Error", f"Invalid input in row {i+1}.")
                return

    def multi_gaussian(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            y += gaussian(x, params[i], params[i+1], params[i+2])
        return y

    try:
        popt, pcov = curve_fit(
            multi_gaussian, x_data, y_data, p0=params,
            bounds=(bounds_lower, bounds_upper)
        )
        errors = np.sqrt(np.diag(pcov))
        
        for i in range(10):
            if not check_vars[i].get():
                amplitude_entries[i].delete(0, tk.END)
                amplitude_entries[i].insert(0, f"{popt[i*3]:.4f}")
                center_entries[i].delete(0, tk.END)
                center_entries[i].insert(0, f"{popt[i*3+1]:.4f}")
                fwhm_entries[i].delete(0, tk.END)
                fwhm_entries[i].insert(0, f"{popt[i*3+2]:.4f}")
                
                amplitude_errors[i].delete(0, tk.END)
                amplitude_errors[i].insert(0, f"{errors[i*3]:.4f}")
                center_errors[i].delete(0, tk.END)
                center_errors[i].insert(0, f"{errors[i*3+1]:.4f}")
                fwhm_errors[i].delete(0, tk.END)
                fwhm_errors[i].insert(0, f"{errors[i*3+2]:.4f}")

        plot_results(x_data, y_data, popt)

    except Exception as e:
        messagebox.showerror("Error", f"Fitting failed: {e}")

def plot_results(x, y, params):
    ax.clear()
    ax.plot(x, y, label="Data", color="blue")
    fit_y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        fit_y += gaussian(x, params[i], params[i+1], params[i+2])
        ax.plot(x, gaussian(x, params[i], params[i+1], params[i+2]), linestyle="--")
    ax.plot(x, fit_y, label="Fit", color="red")
    ax.legend()
    canvas.draw()

def load_file():
    global x_data, y_data
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not filepath:
        return

    try:
        data = np.loadtxt(filepath, delimiter=",")
        x_data = data[:, 0]
        y_data = data[:, 1]
        
        ax.clear()
        ax.plot(x_data, y_data, label="Data")
        ax.legend()
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")

def toggle_entries(row):
    state = "normal" if not check_vars[row].get() else "disabled"
    amplitude_entries[row].config(state=state)
    center_entries[row].config(state=state)
    fwhm_entries[row].config(state=state)

root = tk.Tk()
root.title("Fitting Tool")

frame_controls = ttk.Frame(root)
frame_controls.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

btn_load = ttk.Button(frame_controls, text="Load File", command=load_file)
btn_load.grid(row=0, column=0, padx=5, pady=5)

btn_fit = ttk.Button(frame_controls, text="Fit", command=fit_data)
btn_fit.grid(row=0, column=1, padx=5, pady=5)

frame_table = ttk.Frame(frame_controls)
frame_table.grid(row=1, column=0, columnspan=2, pady=10)

headers = ["Amplitude", "Center", "FWHM"]
for col, header in enumerate(headers):
    ttk.Label(frame_table, text=header).grid(row=0, column=col)
    ttk.Label(frame_table, text=f"Error ({header})").grid(row=0, column=col+3)

amplitude_entries = []
center_entries = []
fwhm_entries = []
amplitude_errors = []
center_errors = []
fwhm_errors = []
check_vars = []

for i in range(10):
    check_var = tk.BooleanVar()
    check_vars.append(check_var)
    check_btn = ttk.Checkbutton(frame_table, variable=check_var, command=lambda r=i: toggle_entries(r))
    check_btn.grid(row=i+1, column=0, sticky="w")

    amp_entry = ttk.Entry(frame_table, width=10)
    amp_entry.grid(row=i+1, column=1)
    amplitude_entries.append(amp_entry)

    cen_entry = ttk.Entry(frame_table, width=10)
    cen_entry.grid(row=i+1, column=2)
    center_entries.append(cen_entry)

    fwhm_entry = ttk.Entry(frame_table, width=10)
    fwhm_entry.grid(row=i+1, column=3)
    fwhm_entries.append(fwhm_entry)

    amp_error = ttk.Entry(frame_table, width=10, state="readonly")
    amp_error.grid(row=i+1, column=4)
    amplitude_errors.append(amp_error)

    cen_error = ttk.Entry(frame_table, width=10, state="readonly")
    cen_error.grid(row=i+1, column=5)
    center_errors.append(cen_error)

    fwhm_error = ttk.Entry(frame_table, width=10, state="readonly")
    fwhm_error.grid(row=i+1, column=6)
    fwhm_errors.append(fwhm_error)

frame_plot = ttk.Frame(root)
frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

x_data, y_data = np.array([]), np.array([])
root.mainloop()


