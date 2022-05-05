import re
from math import sqrt
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pickle
st_words = stopwords.words('english')


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def name_process(text):
    '''THIS FUNCTION IS USED TO PREPROCESS THE NAME FEATURE'''
    text = decontracted(text)
    text = re.sub("[^A-Za-z0-9 ]", "", text)  # REMOVE EVERYTHING EXCEPT THE PROVIDED CHARACTERS
    text = text.lower()  # CONVERT TO LOWER CASE
    text = " ".join([i for i in text.split() if i not in st_words])
    if len(text) == 0:
        text = "missing"
    return text  # RETURN THE OUTPUT TEXT


# preprocessing item description
stopwords = stopwords.words('english')


def preprocess_desc(text_col):
    preprocessed_descs = []
    for sentence in tqdm(text_col.values):
        sent = sentence.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent = ' '.join(e for e in sent.split() if e not in stopwords)
        preprocessed_descs.append(sent.lower().strip())
    return preprocessed_descs


def category_name_preprocessing(text):
    text = re.sub("[^A-Za-z0-9/ ]", "", text)  # REMOVING ALL THE TEXT EXCEPT THE GIVEN CHARACTERS
    text = re.sub("s ", " ", text)  # REMOVING  "s" AT THE END OF THE WORD
    text = re.sub("s/", "/", text)  # REMOVING  "s" AT THE END OF THE WORD
    text = re.sub("  ", " ", text)  # REMOVING ONE SPACE WHERE TWO SPACES ARE PRESENT
    text = text.lower()  # CONVERTING THE TEXT TO LOWER CASE
    return text  # RETURNING THE PROCESSED TEXT


def brand_process(text):
    text = re.sub("[^A-Za-z0-9 ]", "", text)  # REMOVE EVERYTHING EXCEPT THE PROVIDED CHARACTERS
    text = text.lower()  # CONVERT TO LOWER CASE
    return text


def read_csv():
    train = pd.read_csv("train.tsv", delimiter="\t", index_col=["train_id"], na_values=[""])
    test = pd.read_csv("test.tsv", delimiter="\t", index_col=["test_id"])
    return train, test


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def preprocess_data(train):
    train["category_name"] = train["category_name"].fillna("missing")

    # HERE WE ARE PREPROCESSING THE TEXT IN "category_name"
    train["category_name"] = train.category_name.apply(category_name_preprocessing)
    # FORMING A COLUMN "Tier_1"
    train["gen_cat"] = train.category_name.apply(lambda x: x.split("/")[0] if len(x.split("/")) >= 1 else "missing")

    # FORMING A COLUMN "Tier_2"
    train["sub1_cat"] = train.category_name.apply(lambda x: x.split("/")[1] if len(x.split("/")) > 1 else "missing")

    # FORMING A COLUMN "Tier_3"
    train["sub2_cat"] = train.category_name.apply(lambda x: x.split("/")[2] if len(x.split("/")) > 1 else "missing")

    train.drop('category_name', axis=1, inplace=True)

    train['processed_name'] = train.name.apply(name_process)

    train['item_description'] = train['item_description'].fillna("missing")
    train['preprocessed_description'] = preprocess_desc(train['item_description'])

    brand_score = dict(train[train.brand_name.notnull()]["brand_name"].apply(brand_process).value_counts())
    processed_brand_name = []  # storing the barand name after preprocessing
    for index, i in tqdm(train.iterrows()):  # for each row in the dataset

        if pd.isnull(i.brand_name):  # if the brand name isnull we follow this

            words = i.processed_name.split()  # we will split the name for that datapoint

            score = []  # this variable stores the score for each word that we calculated above
            for j in words:  # for each word
                if j in brand_score.keys():  # if the words in name is present in the keys of brand score dict
                    score.append(brand_score[j])  # take the score from the dict and append in the score variable
                else:  # if the word is not a brand name append -1
                    score.append(-1)
            # once we get the scores for all the words in the name the word with maximum score woulb be the brand name
            if max(
                    score) > 0:  # if the maximum score is greater than 0 then it contains a brand name so we append the brand name
                processed_brand_name.append(words[score.index(max(score))])
            else:  # if maximum value is less than 0 then it means no brand name was found so "missing" is appended
                processed_brand_name.append("missing")

        else:  # if the brand_name is not null we follow this
            processed_brand_name.append(brand_process(i.brand_name))

    train['brand_name'] = processed_brand_name

    return train


def encodingCategoricalFeatures(column):
    uniq_val = column.unique()

    unique_dict = {}
    encode_arr = []
    for i in range(0, len(uniq_val), 1):
        unique_dict[uniq_val[i]] = i

    for j in (column):
        value = unique_dict.get(j)
        encode_arr.append(value)
    return unique_dict, encode_arr


def main():
    train, testDf = read_csv()
    trained = preprocess_data(train)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    trained['price'] = trained['price'].fillna(0)
    y_tr = trained['price'].astype(int)
    trained.drop('price', axis=1, inplace=True)
    brands = trained['brand_name'].unique()
    f = open("brands.txt", "w+")
    f.write('[ ')
    for brand in brands:
        f.write('\''+brand+'\', ')
    f.write(']')
    f.close()
    # np.savetxt("brands.txt",np.array(brands), delimiter=", ")
    print(brands)
    brand_encode,trained['brand_name'] = encodingCategoricalFeatures(trained['brand_name'])
    gen_cat_encode, trained['gen_cat'] = encodingCategoricalFeatures(trained['gen_cat'])
    sub1_cat_encode,trained['sub1_cat'] = encodingCategoricalFeatures(trained['sub1_cat'])
    sub2_cat_encode, trained['sub2_cat'] = encodingCategoricalFeatures(trained['sub2_cat'])

    # pickle.dump(brand_encode, open('brand_encode', 'ab'))
    # pickle.dump(gen_cat_encode, open('gen_cat_encode', 'ab'))
    # pickle.dump(sub1_cat_encode, open('sub1_cat_encode', 'ab'))
    # pickle.dump(sub2_cat_encode, open('sub2_cat_encode', 'ab'))

    x_train, x_test, y_train, y_test = train_test_split(trained, y_tr, test_size=0.33, random_state=42)

    X_train = x_train[['shipping','item_condition_id','brand_name','gen_cat','sub1_cat','sub2_cat']].copy()
    X_cv = x_test[['shipping','item_condition_id','brand_name','gen_cat','sub1_cat','sub2_cat']].copy()
    print('xtrain shape=',x_train.shape)
    print('ytrain shape=', y_train.shape)
    print('length of x_train = ', len(x_train))
    print('length of y_train = ', len(y_train))

    lgb_model = LGBMRegressor(subsample=0.9)

    params = {'learning_rate': uniform(0, 1),
              'n_estimators': randint(200, 1500),
              'num_leaves': randint(20, 200),
              'max_depth': randint(2, 15),
              'min_child_weight': uniform(0, 2),
              'colsample_bytree': uniform(0, 1),
              }
    lgb_random = RandomizedSearchCV(lgb_model, param_distributions=params, n_iter=10, cv=3, random_state=42,
                                    scoring='neg_root_mean_squared_error', verbose=10, return_train_score=True)
    lgb_random = lgb_random.fit(X_train, y_train)

    best_params = lgb_random.best_params_
    print(best_params)

    model = LGBMRegressor(**best_params, subsample=0.9, random_state=42, n_jobs=-1)
    model.fit(X_train,y_train)
    lgb_preds_tr = model.predict(X_train)
    lgb_preds_cv = model.predict(X_cv)

    print('Train RMSLE: ', sqrt(mse(y_train, lgb_preds_tr)))
    print('CV RMSLE: ', sqrt(mse(y_test,lgb_preds_cv)))

    # pickle.dump(model, open('finalized_model.sav', 'wb'))

main()
