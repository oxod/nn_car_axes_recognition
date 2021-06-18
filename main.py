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


def load_data(file_name, output_vector_size):
    with open(file_name) as json_file:
        data = json.load(json_file)
        data = data['WeightingItem']

    tractors = [xi for xi in data if xi['machine'] == 'Tractor']
    trailers = [xi for xi in data if xi['machine'] != 'Tractor']

    samples = []

    # concatenate linked tractors and trailers
    # convert trailer type into index vector + tractor length as output sample vector
    for trac in tractors:
        trail = next((t for t in trailers if t['id'] == trac['trailer_link']), None)
        d_lorry = [float(s_dist) for s_dist in trac['dist'].split(' ')]
        d_lorry_length = len(d_lorry)

        s_trail = None
        if trail is not None:
            s_trail = trail['machine']
            for t_dist in trail['dist'].split(' '):
                d_lorry.append(int(t_dist))

        y_vector = tf.keras.utils.to_categorical(get_lorry_type(s_trail), output_vector_size)
        y_vector[-1] = d_lorry_length

        x, y = d_lorry, y_vector
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
    y_length = max_input_length + output_vector_size
    samples_transformed = np.empty((m, y_length))
    for i in range(m):
        for j in range(y_length):
            if j < max_input_length:
                samples_transformed[i][j] = X_transformed[i][j]
            else:
                samples_transformed[i][j] = samples[i][1][j - max_input_length]

    return samples_transformed


def create_model(input_shape, output_vector_size):
    X_input = tf.keras.layers.Input(input_shape)

    X = tf.keras.layers.Dense(10)(X_input)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(20)(X)
    X = tf.keras.layers.BatchNormalization(axis=1)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Dense(output_vector_size)(X)
    X = tf.keras.layers.Activation('softmax')(X)

    model = tf.keras.models.Model(inputs=X_input, outputs=X)
    model.summary()
    return model


def main():
    # 3  binary values for each category softmax output + 1 delimiter of tractor-n-trailer
    output_vector_size = 4

    result = load_data('car_axes_stat.json', output_vector_size)
    X = result[:, 0:-output_vector_size]
    Y = result[:, -output_vector_size:]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = create_model(X_train.shape[1], output_vector_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1000, batch_size=32)
    loss, metrics = model.evaluate(X_test, Y_test, verbose=1)
    print(loss)
    print(metrics)

main()

