import pickle
import re


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


def brand_encode(text):
    encoding = pickle.load(open('brand_encode', 'rb'))

    if text in encoding:
        return encoding[text]
    else:
        return 0


def gen_cat_encode(text):
    encoding = pickle.load(open('gen_cat_encode', 'rb'))

    if text in encoding:
        return encoding[text]
    else:
        return 0


def sub1_cat_encode(text):
    encoding = pickle.load(open('sub1_cat_encode', 'rb'))

    if text in encoding:
        return encoding[text]
    else:
        return 0


def sub2_cat_encode(text):
    encoding = pickle.load(open('sub2_cat_encode', 'rb'))

    if text in encoding:
        return encoding[text]
    else:
        return 0


def getPrediction(shipping, item_condition, brand_name, gen_cat, sub1_cat, sub2_cat):
    print(shipping, item_condition, brand_process(brand_name), category_name_preprocessing(gen_cat), category_name_preprocessing(sub1_cat), category_name_preprocessing(sub2_cat))
    brandN = brand_process(brand_name)
    branName = brand_encode(brandN)
    genCat = gen_cat_encode(category_name_preprocessing(gen_cat))
    sub1Cat = sub1_cat_encode(category_name_preprocessing(sub1_cat))
    sub2Cat = sub2_cat_encode(category_name_preprocessing(sub2_cat))
    X = [[shipping, item_condition, branName, genCat, sub1Cat, sub2Cat]]
    print(X)
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

    price = loaded_model.predict(X)
    return price
