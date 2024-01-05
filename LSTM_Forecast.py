import pandas as pd
import numpy as np
import datetime as datetime
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import openpyxl

from numpy import concatenate
from datetime import timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, mean_absolute_error, mean_squared_error 
from sklearn.preprocessing import MinMaxScaler

# Membuat fungsi untuk membuat data yang akan dijadikan input pada model
def create_sequences(df2, seq_length):
    df_np = df2.to_numpy()
    X, y = [], []
    for i in range(len(df_np) - seq_length):
        X.append(df_np[i:i+seq_length])
        y.append(df_np[i+seq_length])
    return np.array(X), np.array(y)

# Membuat fungsi untuk memprediksi data yang kedepannya
def predict_data(df2, seq_length, n_features, jumlah_periode, model):
    x_input = np.array(df2[len(df2) - seq_length:])
    data_input = list(x_input)
    list_output = []
    o = 0
    #print(x_input)
    
    while(o < jumlah_periode):
        
        if(len(data_input) > seq_length):
            x_input = np.array(data_input[1:])
            x_input = x_input.reshape((1, seq_length, n_features))
            yhat = model.predict(x_input, verbose = 0)
            data_input.append(yhat[0][0])
            data_input = data_input[1:]
            list_output.append(yhat[0][0])
            
            o = o+1
            
        else:
            x_input = x_input.reshape((1, seq_length, n_features))
            yhat = model.predict(x_input, verbose = 0)
            data_input.append(yhat[0][0])
            list_output.append(yhat[0][0])
            
            o = o+1
        
    return list_output

# Membaca data awla
chunk = pd.read_csv('dataset2.csv', chunksize = 1000)
df = pd.concat(chunk)
df = df[["READ_DATE", "KVARH_EXPORT_TOTAL"]]
df3 = df[["KVARH_EXPORT_TOTAL"]]
date_target = df[["READ_DATE"]]
df["READ_DATE"] = pd.to_datetime(df["READ_DATE"])

# Menyiapkan data untuk di preprocess
df = df.groupby(pd.Grouper(key = "READ_DATE", freq = "15T")).mean()
group_names = list(df.index.strftime('%Y-%m-%d %H:%M:%S'))
group_names = pd.to_datetime(group_names)
df_complete = np.asarray(group_names)
df_complete = pd.DataFrame(df_complete)
nilai = df["KVARH_EXPORT_TOTAL"]
nilai_to_np = np.asarray(nilai)

first_index = group_names.min()
first_index += timedelta(weeks=5)
first_index_bfr = group_names.min()
last_index = group_names.max()

# Membuat data mulai dari 1 bulan setelah nilai pertama
cleansed_df = df[(group_names >= first_index) & (group_names <= last_index)]
cleansed_df2 = df[(group_names >= first_index) & (group_names <= last_index)]
df2 = np.asarray(cleansed_df[["KVARH_EXPORT_TOTAL"]]).flatten()
df2 = pd.DataFrame(data={'KVARH_EXPORT_TOTAL' : df2})
nilai2 = df2["KVARH_EXPORT_TOTAL"]
print(df)
print(cleansed_df)

# Membuat data mulai dari awal hingga 1 bulan setelahnya
cleansed_df_bfr = df[(group_names >= first_index_bfr) & (group_names <= first_index)]
df_bfr = np.asarray(cleansed_df_bfr[["KVARH_EXPORT_TOTAL"]]).flatten()
df_bfr = pd.DataFrame(data={'KVARH_EXPORT_TOTAL' : df_bfr})
nilai_bfr = df_bfr["KVARH_EXPORT_TOTAL"]
#print(len(df_bfr))

cleansed_df = list(cleansed_df.index.strftime('%Y-%m-%d %H:%M:%S'))
cleansed_df = pd.to_datetime(cleansed_df)

seq_length = 15
n_features = 1
df_input = df3["KVARH_EXPORT_TOTAL"]
X, y = create_sequences(df_input, seq_length)

scaler = MinMaxScaler(feature_range=(0, 1))
train_size = int(len(df2) * 0.8)
val_size = int(len(df2) * 0.1) + train_size
test_size = len(df2) - train_size + val_size

# Menyiapkan data untuk dimasukkan ke dalam model
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]

# Membuat model LSTM
model1 = Sequential([
    InputLayer((seq_length, n_features)),
    LSTM(20, return_sequences = True),
    Dropout(0.4),
    LSTM(20, return_sequences = True),
    Dropout(0.4),
    LSTM(20, return_sequences = True),
    Dropout(0.4),
    LSTM(10),
    Dropout(0.4),
    Dense(8, activation = 'relu'),
    Dense(1, activation = 'linear')
])

# TRAINING MODELNYA DI LUAT FOR LOOP?

# for i in range(len(nilai2)):
#     if np.isnan(nilai2[i]):
#         print("Data kosong pada indeks ke - ", i)
#         index_awal_input = int(len(df_bfr))
        
#         # Membuat dataframe untuk 7 data sebelum data hilang ditemukan
#         max_time_index = df['KVARH_EXPORT_TOTAL'].isna().idxmax()
#         max_time = cleansed_df[i]
#         min_time = group_names.min()
#         filtered_data = df[(group_names >= min_time) & (group_names <= max_time)]
#         input1 = filtered_data.tail(11).dropna()
#         nilai_input1 = input1['KVARH_EXPORT_TOTAL']
        
#         # Membuat dataframe untuk data 1 minggu yang lalu
#         input2 = []
#         min_time2 = min_time
#         max_time2 = max_time - timedelta(days=7)
#         input2_df = df[(group_names >= min_time2) & (group_names < max_time2)]
#         input2 = input2_df.tail(10)
#         nilai_input2 = input2['KVARH_EXPORT_TOTAL']
        
#         # Membuat dataframe untuk data 1 bulan yang lalu
#         input3 = []
#         min_time3 = min_time2
#         max_time3 = max_time - timedelta(weeks=4)
#         input3_df = df[(group_names >= min_time3) & (group_names < max_time3)]
#         input3 = input3_df.tail(10)
#         nilai_input3 = input3['KVARH_EXPORT_TOTAL']
        
#         # Membuat dataframe untuk data 2 minggu yang lalu
#         input4 = []
#         min_time4 = min_time3
#         max_time4 = max_time2 - timedelta(weeks=1)
#         input4_df = df[(group_names >= min_time4) & (group_names < max_time4)]
#         input4 = input4_df.tail(10)
#         nilai_input4 = input4['KVARH_EXPORT_TOTAL']
        
#         # Membuat dataframe untuk data 3 minggu yang lalu
#         input5 = []
#         min_time5 = min_time4
#         max_time5 = max_time4 - timedelta(weeks=1)
#         input5_df = df[(group_names >= min_time5) & (group_names < max_time5)]
#         input5 = input5_df.tail(10)
#         nilai_input5 = input5['KVARH_EXPORT_TOTAL']
        
#         #print(input1, input2, input3, input4, input5)
        
#         if input1.isnull().values.any() or input2.isnull().values.any() or input3.isnull().values.any() or input4.isnull().values.any() or input5.isnull().values.any():
            
#             print("terdapat data yang kosong di input\n")
#             continue
        
#         else:
            
#             # Gabungkan input
#             combined_input = np.concatenate((input3, input5, input4, input2, input1), axis=0)
    
#             # Bentuk ulang data input
#             combined_input = combined_input.reshape((combined_input.shape[0], combined_input.shape[1]))
            
#             cp1 = ModelCheckpoint('model1/', save_best_only=True)
#             model1.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate=0.001), metrics = [RootMeanSquaredError(), 'acc'])
            
#             # Membuat earlystopping callback
#             early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            
#             history = model1.fit(X_train, y_train, epochs = 50, batch_size = 64, validation_data=(X_val, y_val), callbacks=[early_stopping])
            
#             trainScore = model1.evaluate(X_train, y_train, verbose=0)
#             trainScore = trainScore[0]
#             print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
#             testScore = model1.evaluate(X_test, y_test, verbose=0)
#             testScore = testScore[0]
#             print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
            
#             # Simpan model yang sudah dilatih
#             model1.save("LSTM1")
#             model_LSTM = tf.keras.models.load_model("LSTM1")
            
#             ###################################################################
#             # Codingan untuk mengetes model
#             ###################################################################
            
#             # Mencoba model dengan data X_test
#             test_data_list = []
#             for p in range(len(combined_input)):
#                 test_data_list.append(combined_input[p][0])
                
#             predicted_values = model1.predict(combined_input).flatten()

#             train_predict =predicted_values
#             combined_input_test = combined_input.flatten()
#             y_train = y_train.flatten()
#             train_result = pd.DataFrame(data = {'Actual':list(test_data_list), 'Test Prediction':list(train_predict)})
#             print(train_result)
            
#             # Mengexport nilai tes prediksi ke file excel
#             train_result.to_excel('Hasil_prediksi_test_data.xlsx')
            
#             plt.plot(train_result)
#             plt.show()
            
#             ###################################################################
#             # Codingan untuk memprediksi nilai selanjutnya
#             ###################################################################
            
#             # seq_length = seq_length
#             # n_features = n_features
#             # jumlah_periode = 1
#             # model = model1
            
            
#             # # Prediksi data yang hilang
#             # d_input = []
            
#             # for m in range(len(combined_input)):
#             #     d_input.append(combined_input[m][0])
#             # missing_data_pred = predict_data(d_input, seq_length, n_features, jumlah_periode, model)#.flatten()
#             # lastest_data = combined_input[len(combined_input) - 16:].tolist()    
            
#             # prediksi_future = pd.DataFrame(missing_data_pred, columns = ['Prediksi data'])
#             # print(prediksi_future)
            
#             # if np.isnan(nilai[i]):
#             #     continue
            
#             # else:
#             #     break
            
#             print("data lengkap")
#         break
    
#     else:
#         continue