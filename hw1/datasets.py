import pandas as pd


def family_dataframe(directory):
    return pd.read_csv("{}/sha256_family.csv".format(directory))


def feature_dataframe(sha256s, directory, keys):
    # Create a list containing a DataFrame for each apk with its features
    features = []
    for i, sha256 in enumerate(sha256s):
        try:
            single_apk = pd.read_csv("{}/feature_vectors/{}".format(directory, sha256), sep="::", header=None,
                                     names=["feature_type", "feature_value"])

        # There is a malformed string, i.e. that contains "::"
        except pd.errors.ParserError:
            single_apk = pd.DataFrame(columns=["feature_type", "feature_value"])
            with open("{}/feature_vectors/{}".format(directory, sha256)) as fin:
                for row in fin:
                    split_row = row.split("::", 1)
                    single_apk.append({'feature_type': split_row[0], 'feature_value': split_row[1]},
                                      ignore_index=True)

        single_apk = single_apk.dropna().groupby("feature_type")["feature_value"].apply(list).to_frame().transpose()
        single_apk["sha256"] = [sha256]
        features.append(single_apk)

        print("{} {}".format(i, sha256))

    # Join list elements in a single DataFrame
    features = pd.concat(features, ignore_index=True, sort=False).set_index("sha256")

    # Select only relevant features
    if keys:
        features = features[keys]

    # Replace NaNs with empty list
    # Can't use fillna() because it doesn't accept lists
    # Can't use regex because list is an unhashable type
    for col in features:
        for row in features.loc[features[col].isna(), col].index:
            features[col][row] = []

    return features


def one_hot_encode(enc, features, keys):
    binaries = []
    for key in keys:
        binaries.append(pd.DataFrame(enc.fit_transform(features[key])))
    return pd.concat(binaries, axis=1)
