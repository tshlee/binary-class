import numpy as np
import tensorflow as tf
import pandas as pd

class Model:
  FEATURE_COLS = ['operational_setting_1',
                  'operational_setting_2',
                  'sensor_measurement_1',
                  'sensor_measurement_3',
                  'sensor_measurement_4',
                  'sensor_measurement_5',
                  'sensor_measurement_6',
                  'sensor_measurement_7',
                  'sensor_measurement_8',
                  'sensor_measurement_9',
                  'sensor_measurement_10',
                  'sensor_measurement_11',
                  'sensor_measurement_12',
                  'sensor_measurement_14',
                  'sensor_measurement_15',
                  'sensor_measurement_16',
                  'sensor_measurement_17',
                  'sensor_measurement_18',
                  'sensor_measurement_19',
                  'sensor_measurement_20',
                  'sensor_measurement_21']
  CATEGORY_COL = 'operational_setting_3'
  N_FEATURES = len(FEATURE_COLS)

  def __init__(self, n_days=40, window_size=1):
    """
    Arguments:
    n_days -- unit goes within the next n_days
    window_size -- hyperparameter specifying time window over which features are smoothed
    """
    self.n_days = n_days
    self.window_size = window_size

    #Model is either trained or loaded from disk
    self.model_path = 'model_days%d_window%d' % (self.n_days, self.window_size)
    self.model = None
    self.train_mean = None
    self.train_std = None

  def load(self):
    """Load a saved model from disk
    """
    self.model = tf.keras.models.load_model(self.model_path)
    print('Loaded model "%s"' % (self.model_path))
    self.train_mean = np.array([9.70202349e+00, 2.29616209e-01, 5.00345631e+02, 1.51754372e+03,
          1.31910707e+03, 1.19850011e+01, 1.75194209e+01, 4.47960228e+02,
          2.32481863e+03, 8.84392670e+03, 1.21833553e+00, 4.54495159e+01,
          4.20671308e+02, 8.11144209e+03, 8.71869672e+00, 2.72407733e-02,
          3.73920087e+02, 2.32419229e+03, 9.90961287e+01, 3.17659047e+01,
          1.90652870e+01])
    self.train_std = np.array([1.50609706e+01, 3.42232953e-01, 2.79109893e+01, 1.06614728e+02,
          1.24234338e+02, 3.94001103e+00, 5.99520178e+00, 1.61437996e+02,
          1.19738417e+02, 3.38247625e+02, 1.29299744e-01, 3.03371375e+00,
          1.52807227e+02, 6.71983315e+01, 6.43367397e-01, 4.47027125e-03,
          2.80358269e+01, 1.20553206e+02, 3.57870796e+00, 1.09068265e+01,
          6.53765325e+00])

  def predict(self, df_raw, unit_number, is_sorted=False):
    """Predict if a particular unit will go within the next n_day days.

    Arguments:
    df_raw -- contains expected feature columns and time_stamp if is_sorted=False
    unit_number -- unit_number for which predictions are made
    is_sorted -- if False, it will sort df_raw by time_stamp
    """
    X_raws = self.preprocess_predict(df_raw, unit_number, is_sorted)
    count = 0
    for X_raw in X_raws:
      count += 1
      X = self.features(X_raw, self.window_size)
      p = self.model.predict(X)
      print(count, 'Probability:', p.item(0), 'Prediction:', 1 if p>0.5 else 0)
  
  def train(self, df_raw):
    """Train and save the model to disk.

    Arguments:
    df_raw -- containing the expected feature columns and status, timestamp, etc
    """
    result = self.preprocess_train(df_raw)
    X = np.zeros((0, self.N_FEATURES))
    y = np.zeros((0, 1))
    for pair in result.values():
      X_pos = self.features(pair['pos'], self.window_size)
      X_neg = self.features(pair['neg'], self.window_size)
      X = np.concatenate((X, X_pos, X_neg), axis=0)
      y = np.concatenate((y, np.ones((np.size(X_pos,0), 1)),
                          np.zeros((np.size(X_neg,0), 1))), axis=0)

    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Dense(1, input_dim=self.N_FEATURES, activation='sigmoid'))
    self.model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])
    self.history = self.model.fit(x=X, y=y, batch_size=None, epochs=400, validation_split=0.0, verbose=1)
    self.model.save(self.model_path)
    print('saved "%s"' % self.model_path)
    #print('train_mean', self.train_mean)
    #print('train_std', self.train_std)

  def features(self, X_raw, window_size):
    """Compute smoothing window feature.  Computes mean over the window.

    Arguments:
    X_raw -- (N + window_size - 1) x N_FEATURES matrix of raw features
    window_size -- size of the interval to smooth over by computing mean

    Returns:
    X -- N x N_FEATURES matrix
    """
    X_raw = (X_raw - self.train_mean) / self.train_std[None, :]
    n_examples = np.size(X_raw,0) - window_size + 1
    X = np.zeros((n_examples, self.N_FEATURES))
    for i in range(n_examples):
      X[i,:] = np.mean(X_raw[i:i+window_size,:], axis=0)
    return X

  def preprocess_predict(self, df_raw, unit_number, is_sorted=False, chunk=10):
    """Prepare data for prediction.  Function is a generator object.

    Arguments:
    df_raw -- contains expected feature columns and time_stamp if is_sorted=False
    unit_number -- unit_number for which predictions are made
    is_sorted -- if False, it will sort df_raw by time_stamp
    chunk -- number of examples at each yield
    """
    if not is_sorted:
      df = df_raw.sort_values(by='time_stamp',ascending=True)
      df = df.reset_index(drop=True)
      df = df[:-df['time_stamp'].isna().sum()]

    #Restrict to unit_number
    df_u = df[df['unit_number']==unit_number]
    df_u = df_u[self.FEATURE_COLS]

    #Handle NaN values
    for i in range(self.N_FEATURES):
      col = self.FEATURE_COLS[i]
      df_u[col] = df_u[col].fillna(self.train_mean.item(i))

    #Yield back examples, ideally chunked yields
    for i in range(self.window_size - 1, len(df_u)):
      yield df_u[(i - self.window_size + 1):(i + 1)].to_numpy()
  
  def preprocess_train(self, df_raw):
    """Prepare data for training.

    Arguments:
    df_raw -- containing the expected feature columns and status, timestamp, etc
    """
    # Sort for windows
    df = df_raw.sort_values(by='time_stamp',ascending=True)
    df = df.reset_index(drop=True)

    # Drop NaN time_stamp at the end
    df = df[:-df['time_stamp'].isna().sum()]  #Only 10/4200 in training

    # Calculate mean and std for normalization
    self.train_mean = df[self.FEATURE_COLS].mean(axis=0).to_numpy()
    self.train_std = df[self.FEATURE_COLS].std(axis=0).to_numpy()

    # Replace NaN with global mean
    for i in range(self.N_FEATURES):
      col = self.FEATURE_COLS[i]
      print('in',col,'replacing',df[col].isna().sum(),'nan values with',self.train_mean.item(i))
      df[col] = df[col].fillna(self.train_mean.item(i))

    # For each unit, return X_pos and X_neg padded by window_size
    result = {}
    for unit_name in df['unit_number'].unique():
      df_u = df[df['unit_number']==unit_name]
      if df_u['status'].sum() == 0:
        print('skipped',unit_name)
        continue
      df_u = df_u[self.FEATURE_COLS]
      pos_start = -self.n_days
      neg_start = -self.n_days - self.n_days #self.window_size - 1
      neg_step = 1
      result[unit_name] = {
          'pos': df_u[(pos_start - self.window_size + 1):].to_numpy(),
          'neg': df_u[(neg_start - self.window_size + 1):pos_start:neg_step].to_numpy(),
      }
    return result
  
  def preprocess_category_col(self, df, col, create_categories=False):
    """Unused.  Expand a categorical column into required one-hot columns.

    Arguments:
    df -- data
    col -- name of categorical column
    create_categories -- indicates whether to determine and remember categories
    """
    if create_categories:
      self.categories = df[col].unique()  #Determine categories
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col,
                                       columns=self.categories)],axis=1)
    df.drop(['country'],axis=1, inplace=True)