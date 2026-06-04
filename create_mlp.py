import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
# Read the dataset
data = pd.read_csv('KDDTrain+.txt', header=None)

# The label is the second to last column, last column is a difficulty level
X = data.iloc[:, :-2]
y = data.iloc[:, -2]

# The original preprocessing from the project maps specific attacks to 5 main categories
def map_attack_to_category(attack):
    dos_attacks = ['neptune', 'smurf', 'pod', 'teardrop', 'land', 'back', 'apache2', 'udpstorm', 'processtable', 'mailbomb']
    probe_attacks = ['portsweep', 'ipsweep', 'satan', 'nmap', 'mscan', 'saint']
    r2l_attacks = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']
    u2r_attacks = ['buffer_overflow', 'rootkit', 'loadmodule', 'perl', 'sqlattack', 'xterm', 'ps']
    if attack in dos_attacks: return 'DoS'
    elif attack in probe_attacks: return 'Probe'
    elif attack in r2l_attacks: return 'R2L'
    elif attack in u2r_attacks: return 'U2R'
    else: return 'normal'

y = y.apply(map_attack_to_category)

# Get dummies
X = pd.get_dummies(X)

# Ensure it matches the feature columns expected
feature_columns = joblib.load('feature_columns.pkl')
X = X.reindex(columns=feature_columns, fill_value=0)

# Scale
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)

# Encode labels
label_encoder = joblib.load('label_encoder.pkl')
y_encoded = label_encoder.transform(y)

# Build MLP Keras model
model = Sequential([
    Dense(64, activation='relu', input_shape=(120,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
# Train on a subset for speed, or full dataset (it's fast enough on 125k samples)
model.fit(X_scaled, y_encoded, epochs=5, batch_size=256, validation_split=0.1)

print("Saving model to mlp_ids_model.keras...")
model.save("mlp_ids_model.keras")
print("Done!")
