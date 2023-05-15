import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel(r"./benchmark-merge.xlsx")

print(df.iloc[:, [0, 4]])

total = 0
total_time = []

for i in range(0, 4818):
    vID = df.iloc[:, 0][i]
    if i>0:
        prev_vID = df.iloc[:, 0][i-1]
        if vID[0:vID.rfind('#')] == prev_vID[0:prev_vID.rfind('#')]:
            total += df.iloc[:, 4][i]
        else:
            total_time.append(total)
            total = df.iloc[:, 4][i]
    else:
        total = df.iloc[:, 4][i]
    

x = [15, 30, 60]
y_no_chunk = [27.453, 185.830, 1458.690]
y_chunk_5 = [5.390, 13.378, 55.387]
y_chunk_10 = [5.074, 12.540, 60.321]
y_chunk_15 = [13.831, 22.505, 127.359]

plt.plot(x, y_no_chunk, "black", label = "No chunk")
plt.plot(x, y_chunk_5, "green", label = "5 secs chunk")
plt.plot(x, y_chunk_10, "blue", label = "10 secs chunk")
plt.plot(x, y_chunk_15, "red", label = "15 secs chunk")

plt.xlabel('Video length (s)')
plt.ylabel('KFE time (s)')

plt.title('Video setting - 480p30')
plt.legend()

plt.show()