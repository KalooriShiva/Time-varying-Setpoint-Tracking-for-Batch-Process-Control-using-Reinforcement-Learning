
import numpy as np

# Batch time
tf = 80
# number of timesteps
n_tf = 25
# timestep
dt = tf / n_tf
time_list = np.linspace(0, tf, n_tf)
Tref = np.zeros_like(time_list)

# Generating data
for index, value in enumerate(Tref):
    if index <= 10 / dt: Tref[index] = 298 + 0.5*index*dt
    elif index <= 40 / dt: Tref[index] = 303
    elif index <= 50 / dt: Tref[index] = 303 + 0.3*(index*dt - 40)
    elif index <= 60 / dt: Tref[index] = 306
    elif index <= 75 / dt: Tref[index] = 306 - (8/15)*(index*dt - 60)
    else: Tref[index] = 298

data = {
    "time": time_list,
    "Tref": Tref,
}


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(data)
    sns.set_theme()
    sns.lineplot(
        data=df,
        x="time",
        y="Tref",
    )
