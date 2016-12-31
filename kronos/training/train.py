import numpy as np


def training_non_prepare(X, y):
    """
    Dummy function for the prepare option in the generator. It wont do anything with the data.
    It only prints the X,y values.
    """
    return X, y


class Generators(object):
    """
    Training Generators Utilities and Helpers
    """
    @staticmethod
    def slice_input_producer(X, y, prepare=training_non_prepare, batch_size=32, shuffle=False):
        """
        Given two arrays X and y, start an infinite input generator of minibaches of data using
        "prepare" as a function to preprocess the data. If shuffle = True the data will be shuffled.
        If prepare is not defined, the data wont be modified.

        Example:
            X = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            y = ['1', '2', '3', '4', '5', '6', '7', '8']

            def print_single_row(X, y):
                print("x:", X, " y:", y)
                return X, y

            test = Generators.slice_input_producer(X, y, prepare=print_single_row, batch_size=2, shuffle=True)
            for i in range(5):
                print("--"); next(test);

            > --
            > x: a  y: 1
            > x: c  y: 3
            > --
            > x: b  y: 2
            > x: h  y: 8
            > --
            > x: d  y: 4
            > x: f  y: 6
            > --
            > x: g  y: 7
            > x: e  y: 5
            > --
            > x: a  y: 1
            > x: c  y: 3
            > --

        """
        X, y = np.array(X), np.array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Both arrays must have the same size")

        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)

        while True:
            for i in range(int(X.shape[0] / batch_size)):  # batch_size * X = number of data

                excerpt = indices[i * batch_size:(i + 1) * batch_size]
                X_batch, y_batch = X[excerpt], y[excerpt]

                X_sample, y_sample = [None] * batch_size, [None] * batch_size
                for index, (X_single, y_single) in enumerate(zip(X_batch, y_batch)):
                    X_sample[index], y_sample[index] = prepare(X_single, y_single)

                yield np.array(X_sample), np.array(y_sample)
