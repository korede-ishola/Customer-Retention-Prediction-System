def transform_data(X, processor):
    """
    Transforms preprocessed data into model-ready data.

    """

    X = processor.fit_transform(X)
    
    return X