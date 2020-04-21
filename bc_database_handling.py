import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

category_indices = {}


def prep_df(frame, var_dict):
    frame = handle_df(frame, var_dict)
    frame = frame.reset_index(drop=True)
    return frame


def handle_df(df, var_dict):
    df = df.applymap(lambda x: x.split('|') if type(x) == str else x)

    for cat_values in var_dict.values():
        for val in cat_values:
            if val not in var_dict["categorical_vars"] and val not in var_dict["occurence_counts"]:
                df[val] = df[val].apply(lambda x: x if not isinstance(x, list) else x[0] if len(x) else None)
                if val in var_dict["numeric_vars_mean_fill"] + var_dict["numeric_vars_zero_fill"]:
                    df[val] = df[val].apply(lambda x: x.replace('.', '') if type(x) == str else x)
                    df[val] = df[val].apply(lambda x: x.replace(',', '.') if type(x) == str else x)  # amerikanennnn
                    df[val] = pd.to_numeric(df[val])

    df = handle_variables(df)  # na numeric handling omdat preprocessing nodig is.
    return df


def preprocess_df(df, var_dict, url_dict, needed_columns, cat_dict, filled_dict):

    df, var_dict = augment_variables(df, var_dict, url_dict)

    df = fill_variables(df, filled_dict, var_dict)
    for category in var_dict["categorical_vars"]:
        if category in cat_dict.keys():
            try:
                df[category] = df[category].map(lambda x: x if x not in cat_dict[category].keys() else cat_dict[category][x])
            except TypeError:
                #print(df[category][0:50])
                df[category] = df[category].map(lambda x: list(map(lambda y: y if y not in cat_dict[category].keys() else cat_dict[category][y], x))) #pfoe, mapt elements van list naar dict
                #print(df[category][0:50])
        try:
            df[category] = df[category].apply(lambda x: x.split(','))
        except:
            print("not able to split " + category)
        encoder = MultiLabelBinarizer()
        transformed = encoder.fit_transform(df[category])
        encodings = pd.DataFrame(transformed, columns=encoder.classes_)
        new_cols = [str(category) + ' ' + str(sub_cat) for sub_cat in encodings.columns]
        encodings.columns = new_cols
        df = df.join(encodings)
        df = df.drop([category], axis=1)
    for needed_col in needed_columns:
        if needed_col not in df.columns:
            df[needed_col] = 0 #whole column is zero
    df = df[needed_columns] #right order
    return [df, var_dict]


def fill_variables(df, filled_dict, var_dict):
    for val in var_dict["categorical_vars"]: # Dit zet (de meeste) missende categorieen naar de meest voorkomende
        df[val] = df[val].apply(lambda x: ','.join(x) if type(x) == list else x)
        try:
            df[val] = df[val].fillna(filled_dict[val])
        except:  # if the variable has zero fill rate
            df[val] = df[val].fillna(filled_dict[val])
    for val in var_dict["numeric_vars_mean_fill"]:
        try:
            df[val] = df[val].fillna(filled_dict[val])
        except:
            print("no median found " + val)
        df[val] = df[val].fillna(filled_dict[val])  # in case only NaN values, mean is Nan
    for val in var_dict["numeric_vars_zero_fill"]:
        df[val] = df[val].fillna(filled_dict[val])  # dit zet om naar 0
    return df

def map_categories(df, category, cat_dict):
    df[category] = df[category].map(cat_dict)

def handle_variables(df):  # speciale behandelingen voor variabelen. Sommigen staan niet uniform, etc
    df['mortgage'] = df['mortgage'].map(lambda x: x / 1000 if x > 5000 else x)
    df['annual_income_partner'] = df['annual_income_partner'].map(lambda x: x / 1000 if x > 5000 else x)
    df['annual_income'] = df['annual_income'].map(lambda x: x / 1000 if x > 5000 else x)
    df["search_state"] = df["search_state"].apply(lambda x: x if not isinstance(x, list) else x[0] if len(x) else None)
    df["search_state"] = df["search_state"].map({"ja binnen 3 maanden": "Ja", "ja na 3 maanden": "Ja",
                                                 "Ja binnen 3 maanden": "Ja", "Nee nog niet": "Nee",
                                                 "nee": "Nee", "Ja, binnen drie maanden": "Ja",
                                                 "Nee, ik heb geen koopplannen": "Nee"})
    df = df.drop(df[df["age"] < 12].index)
    df = df.drop(df[df["age"] > 110].index)
    df = df.drop(df[df["age_partner"] < 12].index)
    df = df.drop(df[df["age_partner"] > 110].index)
    return df


def other_morph(frame, col_var, thresh):
    other_vars = frame[col_var].value_counts().index[thresh:]
    map_dict = {}
    for var in other_vars:
        map_dict[var] = "Other"
    frame[col_var] = frame[col_var].replace(map_dict)
    return frame
    pass


####################################################################################################
#### Augmentations
##################################################################################################
def augment_variables(frame, var_dict, url_dict):
    frame["Visit_part_of_day"] = time_of_day(frame["visit_time"])
    var_dict["categorical_vars"].append("Visit_part_of_day")

    for var in var_dict["occurence_counts"]:
        frame[var] = frame[var].fillna(0)
        frame[var] = frame[var].map(lambda x : len(x) if isinstance(x, list) else x)

    for var in var_dict["bool_datetimes"]:
        na_cols = frame[var].isna()
        na_cols = na_cols.map(lambda x: 1 if x == False else 0)
        frame[var] = na_cols

    frame["has_partner"] = pd.to_numeric(has_partner(frame))
    var_dict["numeric_vars_mean_fill"].append("has_partner")
    frame["max_income"], frame["total_income"] = pd.to_numeric(income_augments(frame)[0]), pd.to_numeric(
        income_augments(frame)[1])
    var_dict["numeric_vars_mean_fill"].append("max_income")
    var_dict["numeric_vars_mean_fill"].append("total_income")

    frame, var_dict = is_weekend(frame, ["firstvisit", "lastvisit"], var_dict)
    frame, var_dict = morph_zip(frame, var_dict)
    frame, var_dict = morph_city(frame, var_dict)
    frame, var_dict = search_keywords(frame, var_dict)
    frame, var_dict = url_keywords(frame, var_dict)
    subframe, var_dict = transform_urls(frame["url-name"], var_dict, url_dict)
    frame = pd.concat([frame, subframe], axis=1)
    return [frame, var_dict]


def morph_zip(df, var_dict):
    zip_df = pd.read_csv(gitlink + "/zipcode_final.csv")  # , usecols=["Postcode", "Bevolking", "Inkomen", "Gemiddelde_leeftijd"])
    # zip_df.columns = ["zip_code", "population", "income", "avg_age"]
    zip_df["Postcode"] = zip_df["Postcode"].astype(int)
    zip_df["Total_income"] = zip_df["Bevolking"] * zip_df["Inkomen"]

    avg_income_person = zip_df["Total_income"].sum() / zip_df["Bevolking"].sum()
    avg_pop_zip = zip_df["Bevolking"].sum() / len(zip_df["Bevolking"])
    c = 0
    income_population_segment = []
    while c < len(zip_df["Inkomen"]):
        if zip_df["Inkomen"][c] <= avg_income_person:
            if zip_df["Bevolking"][c] <= avg_pop_zip:
                income_population_segment.append("lowinc_lowpop")  # laag inkomen laag bevolking
            else:
                income_population_segment.append("lowinc_highpop")  # laag inkomen hoog bevolking
        else:
            if zip_df["Bevolking"][c] <= avg_pop_zip:
                income_population_segment.append("highinc_lowpop")  # laag inkomen laag bevolking
            else:
                income_population_segment.append("highinc_highpop")  # laag inkomen laag bevolking
        c += 1
    zip_df["income_population_segment"] = income_population_segment
    zip_df.columns = ["zip_code " + x for x in zip_df.columns]  # niet vergeten
    for column in zip_df.columns:
        if column not in ["zip_code Postcode", "zip_code income_population_segment"]:  # alle numeric waardes
            var_dict["numeric_vars_mean_fill"].append(column)
            zip_df[column] = pd.to_numeric(zip_df[column])
        elif column != "zip_code Postcode":
            var_dict["categorical_vars"].append(column)

    df["mr_geo_zipcode"] = df["mr_geo_zipcode"].apply(
        lambda x: int(x) if re.match(r"^[0-9]{4,5}$", str(x)) else None)  # pakt de eerste, moet nog average fixen
    try:
        merged = df.merge(zip_df, left_on="mr_geo_zipcode", right_on="zip_code Postcode",
                          how="left")  # zip_code zip_code as it has been transformed
    except:
        print("Could not merge zipcodes")

    df = merged.drop(["mr_geo_zipcode", "zip_code Postcode"], axis=1)
    return [df, var_dict]


def morph_city(frame, var_dict):
    gemeente_steden = pd.read_csv(gitlink + "/Gemeente_per_woonplaats.csv", delimiter=";", decimal=",")
    gemeente_stad_dict = dict(zip(gemeente_steden["Stad"], gemeente_steden["Gemeente"]))
    steden = frame["mr_geo_city_name"]
    gemeentes = pd.DataFrame({"Gemeente": steden.map(gemeente_stad_dict)})

    gemeente_stats = pd.read_csv(gitlink + "Kerncijfers_per_gemeente.csv", delimiter=";", decimal=",")
    gemeente_stats.columns = ["Gemeente :" + col for col in gemeente_stats.columns]
    merged = gemeentes.merge(gemeente_stats, left_on="Gemeente", right_on="Gemeente :Gemeente", how="left")
    merged = merged.drop(['Gemeente', "Gemeente :Gemeente"], axis=1)

    merged["Gemeente :Inkomen"] = merged["Gemeente :Inkomen"].map(
        lambda x: x.replace(',', '.') if isinstance(x, str) and re.search(r".+,.+", x) else None)
    merged["Gemeente :Inkomen"] = pd.to_numeric(merged["Gemeente :Inkomen"])
    var_dict["numeric_vars_mean_fill"] += list(merged.columns)
    frame = frame.drop("mr_geo_city_name", axis=1)
    return [pd.concat([frame, merged], axis=1), var_dict]


import datetime


def is_weekend(df, date_vars, var_dict):
    for date_var in date_vars:
        datetime_series = df[date_var]
        datetime_series = datetime_series.map(
            lambda x: datetime.datetime.fromtimestamp(int(x) / 1e3))
        is_weekend_series = datetime_series.map(lambda x: 1 if x.weekday() > 4 else 0)
        df[date_var + ' is_weekend'] = is_weekend_series
        var_dict["numeric_vars_mean_fill"].append(date_var + ' is_weekend')
    return [df, var_dict]


import regex as re


def time_of_day(visit_time):
    vals = []
    for list_value in visit_time:
        try:
            value = list_value[0]
            if re.search(r"[5678]\sAM -", value):
                vals.append("Early Morning")
            elif re.search(r"(9|10|11)\sAM -", value):
                vals.append("Late Morning")
            elif re.search(r"(12|1|2)\sPM -", value):
                vals.append("Early Afternoon")
            elif re.search(r"[345]\sPM -", value):
                vals.append("Late Afternoon")
            elif re.search(r"[678]\sPM -", value):
                vals.append("Early Evening")
            elif re.search(r"(9|10|11)\sPM -", value):
                vals.append("Late Evening")
            elif re.search(r"(12|1|2|3|4)\sAM -", value):
                vals.append("Night")
            else:
                vals.append("Unknown")
        except:  # nan
            vals.append(None)
    return vals


def has_partner(frame):
    c = 0
    ages = frame["age"]
    partner_ages = frame['age_partner']
    bools = []
    while c < len(frame):
        if isinstance(ages.iloc[c], int) and isinstance(partner_ages.iloc[c], int):
            bools.append(1)
        else:
            bools.append(0)
        c += 1
    return bools


def search_keywords(frame, var_dict):  # doet niks, niemand zoekt blijkbaar
    user_searches = frame["keywords"]
    bestaande_woning_keywords = ["oversluiten", "verbouw", 'tweede', "extra", "rentevaste periode",
                                 "pensioen", "rentemiddeling"]
    starter_keywords = ["studie", "eerste"]
    volgende_woning_keywords = ["meenemen", "overwaarde", "restschuld", "groter", "verhuizing", "verkoop",
                                "tweede", "overbrug", "2e"]
    bestaande_woning_searches, starter_searches, volgende_woning_searches = np.zeros(len(user_searches)), np.zeros(
        len(user_searches)), np.zeros(len(user_searches))
    c = 0
    for searches in user_searches:  # one can have multiple searches
        if not isinstance(searches, float) and searches is not None:
            for search in searches:
                for keyword in starter_keywords:
                    if keyword in search:
                        starter_searches[c] += 1
                for keyword in bestaande_woning_keywords:
                    if keyword in search:
                        bestaande_woning_searches[c] += 1
                for keyword in volgende_woning_keywords:
                    if keyword in search:
                        volgende_woning_searches[c] += 1
        c += 1
    frame["search bestaande_woning"], frame["search volgende_woning"], frame[
        "search starter"] = bestaande_woning_searches, volgende_woning_searches, starter_searches
    var_dict["numeric_vars_zero_fill"] += ["search bestaande_woning", "search volgende_woning", "search starter"]
    return [frame, var_dict]


def url_keywords(frame, var_dict):  # url has been formatted already
    user_urls = frame["url-name"]
    bestaande_woning_keywords = ["oversluiten", "verbouw", 'tweede', "extra", "rentevaste periode",
                                 "pensioen", "rentemiddeling", "uit elkaar", "scheiden", "hypotheek aanpassen"]
    starter_keywords = ["studie", "eerste woning", "eerste huis", "mezelf wonen", "starter",
                        "alles wat je moet weten als je een huis wilt kopen"]
    volgende_woning_keywords = ["volgende woning", "volgende huis", "overwaarde", "tweede woning", "woning waard"]

    bestaande_woning_urls, starter_urls, volgende_woning_urls = np.zeros(len(user_urls)), np.zeros(
        len(user_urls)), np.zeros(len(user_urls))
    c = 0
    for urls in user_urls:  # one can have multiple searches
        if not isinstance(urls, float) and urls is not None:
            for url in urls:
                for keyword in starter_keywords:
                    if keyword in url:
                        starter_urls[c] += 1
                for keyword in bestaande_woning_keywords:
                    if keyword in url:
                        bestaande_woning_urls[c] += 1
                for keyword in volgende_woning_keywords:
                    if keyword in url:
                        volgende_woning_urls[c] += 1
        c += 1
    frame["url bestaande_woning"], frame["url volgende_woning"], frame[
        "url starter"] = bestaande_woning_urls, volgende_woning_urls, starter_urls
    var_dict["numeric_vars_zero_fill"] += ["search bestaande_woning", "search volgende_woning", "search starter"]
    return [frame, var_dict]


def income_augments(frame):
    income = frame["annual_income"]
    partner_income = frame["annual_income_partner"]
    max_income = []
    total_income = []
    c = 0
    while c < len(income):
        max_income.append(max(income.iloc[c], partner_income.iloc[c]))
        total_income.append(income.iloc[c] + partner_income.iloc[c])
        c += 1
    return [max_income, total_income]


#####
# Clusters

def transform_urls(urls_list, var_dict, url_dict):
    mapped_list = []
    for url_list in urls_list:
        mapped = []
        if isinstance(url_list, list):  # catches Nan
            for url in url_list:
                if url in url_dict.keys():
                    mapped.append(url_dict[url])
        mapped_list.append(mapped)

    df = pd.DataFrame(0, index=np.arange(len(urls_list)), columns=set(url_dict.values()))
    df.columns = ["url_cluster " + str(x) for x in df.columns]
    for x in range(0, len(mapped_list)):
        value_counts = pd.Series(mapped_list[x]).value_counts()
        for index in value_counts.index:
            row = df.loc[x]
            row[index] = value_counts[index]
    var_dict["numeric_vars_zero_fill"] += list(df.columns)

    return [df, var_dict]