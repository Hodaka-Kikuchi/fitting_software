# %% 必須: セル区切り
import numpy as np
import matplotlib.pyplot as plt

# データ生成
x = np.linspace(0, 20, 100)
y = np.cos(x)

# プロット
plt.plot(x, y)
plt.title("Wave")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
# データ生成
x = np.linspace(0, 20, 100)
y = np.tan(x)

# プロット
plt.plot(x, y)
plt.title("Wave")
plt.show()
# %%
