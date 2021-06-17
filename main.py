import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def get_lorry_type(s_trailer_type):
    return {
        None: 0,
        'Trailer': 1,
        'Semitrailer': 2
    }[s_trailer_type]


def convert_to_y_vector(output_length, trailer_type_index, lorry_length):
    r = np.zeros(output_length)
    r[trailer_type_index] = 1
    r[-1] = lorry_length
    return r


def load_data(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        data = data['WeightingItem']

    tractors = [xi for xi in data if xi['machine'] == 'Tractor']
    trailers = [xi for xi in data if xi['machine'] != 'Tractor']

    samples = []

    # concatenate linked tractors and trailers
    # convert trailer type into index vector + tractor length as output sample vector
    output_length = 4 # 3  binary values for each category softmax output + 1 delimiter of tractor-n-trailer
    for trac in tractors:
        trail = next((t for t in trailers if t['id'] == trac['trailer_link']), None)
        d_lorry = [float(s_dist) for s_dist in trac['dist'].split(' ')]
        d_lorry_length = len(d_lorry)

        s_trail = None
        if trail is not None:
            s_trail = trail['machine']
            for t_dist in trail['dist'].split(' '):
                d_lorry.append(int(t_dist))

        x, y = d_lorry, convert_to_y_vector(output_length, get_lorry_type(s_trail), d_lorry_length)
        samples.append((x, y))

    # pad data to have all X samples of the same length
    max_input_length = max(len(s[0]) for s in samples)
    for s in samples:
        pad_to = max_input_length - len(s[0])
        for i in range(pad_to):
            s[0].append(0)

    # normalize input X data
    X = [np.array(s[0]) for s in samples]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_transformed = scaler.transform(X)

    # put all X, Y data into 2D np array
    m = len(samples)
    y_length = max_input_length + output_length
    samples_transformed = np.empty((m, y_length))
    for i in range(m):
        for j in range(y_length):
            if j < max_input_length:
                samples_transformed[i][j] = X_transformed[i][j]
            else:
                samples_transformed[i][j] = samples[i][1][j - max_input_length]

    return samples_transformed


is_training = True

result = load_data('car_axes_stat.json')
output_vector_size = 4
X = result[:, 0:-output_vector_size]
Y = result[:, -output_vector_size:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X = tf.keras.layers.Dense(10)(X)
X = tf.keras.layers.BatchNormalization(axis=1, trainable=is_training)(X)
X = tf.keras.layers.Activation('relu')(X)

X = tf.keras.layers.Dense(20)(X)
X = tf.keras.layers.BatchNormalization(axis=1, trainable=is_training)(X)
X = tf.keras.layers.Activation('relu')(X)

X = tf.keras.layers.Dense(output_vector_size)(X)
X = tf.keras.layers.Activation('softmax')(X)
