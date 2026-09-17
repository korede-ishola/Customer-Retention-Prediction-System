from sklearn.model_selection import train_test_split

def split_data(X, y, test_size, random_state=42):
    """ Splits a dataset into training and test sets. """

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)