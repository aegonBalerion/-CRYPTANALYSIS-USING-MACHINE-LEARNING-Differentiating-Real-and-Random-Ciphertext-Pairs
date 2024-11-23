import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
# Function to encrypt plaintext using simple XOR-based encryption
def simple_cipher_encrypt(plaintext, key):
   return plaintext ^ key


# Function to encrypt plaintext using AES encryption
def aes_encrypt(plaintext, key):
   cipher = AES.new(key, AES.MODE_ECB)
   ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
   return np.frombuffer(ciphertext, dtype=np.uint32)


# Function to generate data samples
def generate_data(num_samples, input_difference, key):
   key_bytes = key.to_bytes(16, byteorder='big')


   # Generate random plaintexts
   plaintexts = np.random.randint(0, 2**32, size=(num_samples, 4), dtype=np.uint32)


   # Create pairs with a fixed input difference
   plaintexts_pair = plaintexts ^ input_difference


   # Encrypt plaintexts using XOR-based encryption
   ciphertexts_real_simple = np.array([
       simple_cipher_encrypt(plaintexts[i], key) for i in range(num_samples)
   ], dtype=np.uint32)
   ciphertexts_pair_real_simple = np.array([
       simple_cipher_encrypt(plaintexts_pair[i], key) for i in range(num_samples)
   ], dtype=np.uint32)


   # Encrypt plaintexts using AES encryption
   ciphertexts_real_aes = np.array([
       aes_encrypt(plaintexts[i].tobytes(), key_bytes) for i in range(num_samples)
   ], dtype=np.uint32)
   ciphertexts_pair_real_aes = np.array([
       aes_encrypt(plaintexts_pair[i].tobytes(), key_bytes) for i in range(num_samples)
   ], dtype=np.uint32)


   # Flatten all ciphertext arrays to ensure they are 2D arrays with the same second dimension
   ciphertexts_real_simple = ciphertexts_real_simple.reshape(num_samples, -1)
   ciphertexts_pair_real_simple = ciphertexts_pair_real_simple.reshape(num_samples, -1)
   ciphertexts_real_aes = ciphertexts_real_aes.reshape(num_samples, -1)
   ciphertexts_pair_real_aes = ciphertexts_pair_real_aes.reshape(num_samples, -1)


   # Generate random ciphertext pairs for comparison
   ciphertexts_random = np.random.randint(0, 2**32, size=(num_samples, 4), dtype=np.uint32)
   ciphertexts_pair_random = np.random.randint(0, 2**32, size=(num_samples, 4), dtype=np.uint32)


   # Concatenate real and random ciphertexts
   data_real = np.hstack((
       ciphertexts_real_simple,
       ciphertexts_pair_real_simple,
       ciphertexts_real_aes,
       ciphertexts_pair_real_aes
   ))
   data_random = np.hstack((ciphertexts_random, ciphertexts_pair_random))


   # Ensure both real and random data have the same dimensions
   data_real = data_real[:, :8]  # Truncate or adjust to the same dimension as data_random
   data = np.vstack((data_real, data_random))


   # Create labels for real and random pairs
   labels_real = np.ones(data_real.shape[0])
   labels_random = np.zeros(data_random.shape[0])


   labels = np.hstack((labels_real, labels_random))


   return data, labels


# Generate dataset
num_samples = 10000
input_difference = np.random.randint(0, 2**32, size=(4,), dtype=np.uint32)
key = np.random.randint(0, 2**32)


data, labels = generate_data(num_samples, input_difference, key)


# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create and train the model
model = create_complex_mlp(X_train.shape[1])
history = model.fit(
   X_train, y_train,
   epochs=10,
   batch_size=32,
   validation_data=(X_test, y_test),
   callbacks=[early_stopping, reduce_lr, model_checkpoint]
)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')