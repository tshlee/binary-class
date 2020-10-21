import pandas as pd
import model  #see model.py

#Train
df_raw = pd.read_csv('/path/to.csv')
m = model.Model(n_days=40, window_size=20)
m.train(df_raw)

#Predict
df_raw = pd.read_csv('/path/to.csv')
m = model.Model(n_days=40, window_size=20)
m.load()
m.predict(df_raw, 'unit_2')

#import matplotlib.pyplot as plt
#plt.plot(m.history.history['loss'], label='loss')
#plt.plot(m.history.history['val_loss'], label='val_loss')
#plt.ylim([0, 1])
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.legend()
#plt.grid(True)