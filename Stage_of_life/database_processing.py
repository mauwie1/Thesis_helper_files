import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import importlib
category_indices = {}
import re
import itertools


def prep_csv_df(csv_files, var_dict, subsample = 1, proportions=[], sizes=[], thousands="."):
    frame_info = construct_frames(csv_files, [item for sublist in var_dict.values() for item in sublist], subsample, proportions, sizes, thousands)
    frame, lengths = frame_info[0], frame_info[1]
    frame = add_class_label(frame, lengths)
    frame = handle_df(frame, var_dict)
    frame = frame.reset_index(drop=True)
    return frame

def prep_bc_df(frame, var_dict):
    frame = handle_df(frame, var_dict, bc=True)
    frame = frame.reset_index(drop=True)
    return frame

def construct_frames(csv_names, variables, subsample, proportions=[], sizes=[], thousands="."):
    lengths = []
    if proportions == [] and sizes== []: #onbekend?
        if thousands=='':
            full_df = pd.read_csv(csv_names[0], sep=',', low_memory=False, usecols=variables)
        else:
            full_df = pd.read_csv(csv_names[0], sep=',', low_memory=False, usecols = variables, thousands=thousands)
        full_df = full_df[:int(subsample*len(full_df))]
        lengths.append(len(full_df)-1)  # -1 omdat de eerste row een 2e header is

        for csv_name in csv_names[1:]:
            if thousands == '':
                csv_df = pd.read_csv(csv_name, sep=',', low_memory=False, usecols=variables)
            else:
                csv_df = pd.read_csv(csv_name, sep=',', low_memory=False, usecols=variables, thousands=thousands)
            csv_df = csv_df[:int(subsample * len(csv_df))]
            lengths.append(len(csv_df) - 1)  # -1 omdat de eerste row een 2e header is
            full_df = pd.concat([full_df, csv_df])
    else:
        max_over_ratio = np.argmax([(proportions[x]*(subsample * sum(sizes))/sizes[x]) for x in range(0, len(sizes))]) #pfoei, argmax van de ratio van sizes vs wanted sizes
        imposed_size = min((sizes[max_over_ratio] / proportions[max_over_ratio]) * proportions[0], subsample*proportions[0]*sum(sizes))
        csv_name = csv_names[0]
        if thousands== '':
            full_df = pd.read_csv(csv_name, sep=',', low_memory=False, usecols=variables, nrows = int(imposed_size))
        else:
            full_df = pd.read_csv(csv_name, sep=',', low_memory=False, usecols=variables, thousands=thousands,nrows=int(imposed_size))
        lengths.append(len(full_df) - 1)
        for x in range(1, len(csv_names)):
            imposed_size = min((sizes[max_over_ratio] / proportions[max_over_ratio]) * proportions[x], subsample*proportions[x]*sum(sizes))
            csv_name = csv_names[x]
            if thousands == '':
                csv_df = pd.read_csv(csv_name, sep=',', low_memory=False, usecols=variables, nrows = int(imposed_size))
            else:
                csv_df = pd.read_csv(csv_name, sep=',', low_memory=False, usecols=variables, thousands=thousands, nrows=int(imposed_size))
            lengths.append(len(csv_df) - 1)  # -1 omdat de eerste row een 2e header is
            full_df = pd.concat([full_df, csv_df])
    full_df = full_df.drop([0])
    return [full_df, lengths]

def handle_df(df, var_dict, bc=False):
    df = df.applymap(lambda x: x.split('|') if type(x) == str else x)

    for cat_values in var_dict.values():
        for val in cat_values:
            if val not in var_dict["categorical_vars_none"]+ var_dict["categorical_vars_median"] + var_dict["occurence_counts"] + var_dict["list_vars"]:
                df[val] = df[val].apply(lambda x: x if not isinstance(x, list) else x[0] if len(x) else None)
                if val in var_dict["numeric_vars_mean_fill"] + var_dict["numeric_vars_zero_fill"]:
                    df[val] = df[val].apply(lambda x: x.replace('.', '') if type(x) == str else x)
                    df[val] = df[val].apply(lambda x: x.replace(',', '.') if type(x) == str else x) #amerikanennnn
                    df[val] = pd.to_numeric(df[val], errors="coerce")

    df = handle_variables(df, bc) # na numeric handling omdat preprocessing nodig is.
    df = stem_urls(df, "recentlyvieweditems")
    return df

def stem_urls(frame, url_var):
    updated_urls = []
    for url_list in frame[url_var]:
        if isinstance(url_list, list):
            new_urls = []
            for url in url_list:
                re_url = re.sub(r'\?.*', '', url)
                re_url = re.sub(r'\/vestigingen\/.*', '/vestigingen/', re_url)
                re_url = re.sub(r'\/nieuwsberichten\/.*', '/nieuwsberichten/', re_url)
                new_urls.append(re_url)
            updated_urls.append(new_urls)
        else:
            updated_urls.append(url_list)
    frame[url_var] = updated_urls
    return frame

def remove_columns(train_x, validation_x, test_x, col_name, sub_cols=True):
    if sub_cols:
        col_name = [x for x in train_x.columns if col_name == x.split()[0]]
    train_x = train_x.drop(col_name, axis=1)
    validation_x = validation_x.drop(col_name, axis=1)
    test_x = test_x.drop(col_name, axis=1)
    return [train_x, validation_x, test_x]


def handle_variables(df, bc):  # speciale behandelingen voor variabelen. Sommigen staan niet uniform, etc
    df['mortgage'] = df['mortgage'].map(lambda x: x / 1000 if x > 5000 else x)
    df['annual_income_partner'] = df['annual_income_partner'].map(lambda x: x * 12 / 1000 if x > 1000 and x < 150000 else x)  # 150000/maand == 1.8 miljoen per jaar, jaja
    df['annual_income_partner'] = df['annual_income_partner'].map( lambda x: x / 1000 if x > 2000 else x)  # mensen die 2 miljoen per jaar verdienen en een hypotheek afsluiten??
    df['annual_income_partner'] = df['annual_income_partner'].map(lambda x: np.nan if x > 2000 else x)  # nonsens variabelen
    df['annual_income'] = df['annual_income'].map(lambda x: x / 1000) #mensen die 2 miljoen per jaar verdienen en een hypotheek afsluiten??
    df['annual_income'] = df['annual_income'].map(lambda x: x*(12/1000) if x > 1000 and x<150000 else x) #150000/maand == 1.8 miljoen per jaar, jaja
    df['annual_income'] = df['annual_income'].map(lambda x: x / 1000 if x > 2000 else x) #mensen die 2 miljoen per jaar verdienen en een hypotheek afsluiten??
    df['annual_income'] = df['annual_income'].map(lambda x: np.nan if x > 2000 else x)  #nonsens variabelen
    df["search_state"] = df["search_state"].apply(lambda x: x if not isinstance(x, list) else x[0] if len(x) else None)
    df["search_state"] = df["search_state"].map({"ja binnen 3 maanden" : "Ja", "ja na 3 maanden":"Ja",
                                                 "Ja binnen 3 maanden" : "Ja", "Nee nog niet": "Nee",
                                                 "nee":"Nee", "Ja, binnen drie maanden": "Ja",
                                                 "Nee, ik heb geen koopplannen" : "Nee"})
    df["age"] = df["age"].map(lambda x: x/10 if x>100 and x<1000 else x)
    df["age_partner"] = df["age_partner"].map(lambda x: x/10 if x>100 and x<1000 else x) #best veel blijkbaar?
    if not bc:
        df = df.drop(df[df["age"] < 12].index)
        df = df.drop(df[df["age"] > 110].index)
        df = df.drop(df[df["age_partner"] < 12].index)
        df = df.drop(df[df["age_partner"] > 110].index)
    return df

def add_class_label(df, seg_amts):
    c=1
    y = np.repeat(0, seg_amts[0])
    while c<len(seg_amts):
        y = np.concatenate([y, np.repeat(c, seg_amts[c])])
        c += 1
    df['y'] = y  # we voegen hem toe zodat we de database kunnen shufflen en de toebehorendheid niet verloren gaat.

    shuffle_df = df.sample(frac=1)  # dit shufflet de database, zodat train en testset een random hoeveelheid van beide klassen hebben.

    return shuffle_df

def list_remove(target, list_var, frame):
    frame[list_var] = frame[list_var].map(lambda x: list(filter(lambda y: y!=target, x)) if isinstance(x, list) else x)
    frame[list_var] = frame[list_var].map(lambda x: np.NaN if x==[] else x) #[] wordt None
    return frame


def resize_train(train_x, train_y, wanted_proportions, smote=False):
    imposed_sizes = [prop * len(train_x) for prop in wanted_proportions]
    actual_sizes = [len(train_y[train_y == y]) for y in sorted(train_y.unique())]
    smallest_ratio = np.argmin([actual_sizes[i] / imposed_sizes[i] for i in range(0, len(wanted_proportions))])
    wanted_sizes = [int(wanted_proportions[i] / wanted_proportions[smallest_ratio] * actual_sizes[smallest_ratio]) for i
                    in range(0, len(wanted_proportions))]
    cl_frame = train_x.join(train_y)
    frame = cl_frame[cl_frame["y"] == 0][0:wanted_sizes[0]]
    for c in range(1, len(wanted_sizes)):
        frame = pd.concat([frame, cl_frame[cl_frame["y"] == c][0:wanted_sizes[c]]])
    res_train_x, res_train_y = frame.drop("y", axis=1), frame["y"]
    return res_train_x, res_train_y


##############################################
# Preprocessing
##############################################

def preprocess_df(df, var_dict, url_dict, url_dict2, gitlink ='',
                  morph_categorical=True, threshold =20, augment = True):
    if augment:
        df, var_dict, augmented_vars = augment_variables(df, var_dict,  url_dict, url_dict2, gitlink)
    else:
        augmented_vars = []
    unfilled_df = df.copy(deep=True)
    df, filled_dict = fill_variables(df, var_dict)
    non_onehot_df = df.copy(deep=True)
    modified_df = df
    for category in var_dict["categorical_vars_median"]+ var_dict["categorical_vars_none"] + var_dict["list_vars"]:
        modified_df = modified_df.drop(category, axis=1)  # prevents faulty indices and allows for references to the column still
    other_dict = {}
    if morph_categorical == True:
        for category in var_dict["categorical_vars_median"]+ var_dict["categorical_vars_none"] + var_dict["list_vars"]:
            if category in var_dict["list_vars"]:
                df, other_dict = other_morph(df, category, threshold, other_dict, is_list=True)
            else:
                if len(df[category].unique()) > threshold:  # anders crash, teveel kolommen
                    df, other_dict = other_morph(df, category, threshold, other_dict)
                try:
                    df[category] = df[category].apply(lambda x: str(x).split(','))
                except:
                    print("not able to split " + category)
            encoder = MultiLabelCounter()
            transformed = encoder.fit_transform(df[category])
            encodings = pd.DataFrame(transformed, columns=encoder.classes_)
            new_cols = []
            for sub_cat in encodings.columns:
                new_cols.append(str(category) + ' ' + str(sub_cat))
            encodings.columns = new_cols
            encodings.index = df.index
            category_indices[category] = range(len(modified_df.columns),
                                               len(modified_df.columns) + len(encodings.columns))
            df = df.join(encodings)
            modified_df = modified_df.join(encodings)
            df = df.drop([category], axis=1)
    return [df, unfilled_df, non_onehot_df, other_dict, filled_dict, var_dict, augmented_vars]


def fill_variables(df, var_dict):
    filled_dict = {}

    for val in var_dict["categorical_vars_median"]+ var_dict["categorical_vars_none"]:  # Dit zet (de meeste) missende categorieen naar de meest voorkomende
        df[val] = df[val].apply(lambda x: ','.join(x) if type(x)==list else x)
        if val in var_dict["categorical_vars_median"]:
            try:
                most_frequent = df[val].value_counts().index[0]
                df[val] = df[val].fillna(most_frequent)
                filled_dict[val] = most_frequent
            except: # if the variable has zero fill rate
                df[val] = df[val].fillna('Empty')
                filled_dict[val] = 'Empty'
        elif val in var_dict["categorical_vars_none"]:
            df[val] = df[val].fillna('Empty')
            filled_dict[val] = 'Empty'
    for val in var_dict["list_vars"]:
        df[val] = df[val].map(lambda x: x if isinstance(x, list) else [])
        filled_dict[val] = []
    for val in var_dict["numeric_vars_mean_fill"]:  # dit zet numerieke waardes naar de gemiddelde waardes
        med = df[val].median()
        if not np.isnan(med):
            df[val] = df[val].fillna(med)
            filled_dict[val] = med
        else:
            df[val] = df[val].fillna(0)
            filled_dict[val] = 0
    for val in var_dict["numeric_vars_zero_fill"]:
        df[val] = df[val].fillna(0)  # dit zet om naar 0
        filled_dict[val] = 0

    for var in var_dict["occurence_counts"]:
        df[var] = df[var].fillna(0)
        df[var] = df[var].map(lambda x: len(x) if isinstance(x, list) else x)

    for var in var_dict["bool_datetimes"]:
        na_cols = df[var].isna()
        na_cols = na_cols.map(lambda x: 1 if x == False else 0)
        df[var] = na_cols
    return [df, filled_dict]


def preprocess_df_bc(df, var_dict, url_dict, url_dict2, gitlink, needed_columns, cat_dict, filled_dict, augment=True):

    if augment:
        df, var_dict, dummy_aug_vars = augment_variables(df, var_dict, url_dict, url_dict2, gitlink)
    df = fill_variables_bc(df, filled_dict, var_dict)
    for category in var_dict["categorical_vars_none"] + var_dict["categorical_vars_median"] + var_dict["list_vars"]:
        if category in cat_dict.keys():
            if category in var_dict["list_vars"]:
                df[category] = df[category].map(lambda x: list(map(lambda y: "Other" if y not in cat_dict[category] else y, x)) if isinstance(x, list) else x) #pfoe, mapt elements van list naar dict
            else:
                    df[category] = df[category].map(lambda x: "Other" if x not in cat_dict[category] else x)
        try:
            df[category] = df[category].map(lambda x: str(x).split(',') if not isinstance(x, list) else x)
        except:
            print("not able to split " + category)
        encoder = MultiLabelCounter()
        transformed = encoder.fit_transform(df[category])
        encodings = pd.DataFrame(transformed, columns=encoder.classes_)
        new_cols = [str(category) + ' ' + str(sub_cat) for sub_cat in encodings.columns]
        encodings.columns = new_cols
        needed_encodings = encodings[encodings.columns&needed_columns]
        encodings=None
        needed_encodings.index = df.index
        df = df.join(needed_encodings)
        df = df.drop([category], axis=1)
    for needed_col in needed_columns:
        if needed_col not in df.columns:
            df[needed_col] = 0 #whole column is zero
    df = df[needed_columns] #right order
    return [df, var_dict]




def fill_variables_bc(df, filled_dict, var_dict):
    for val in var_dict["categorical_vars_median"] + var_dict["categorical_vars_none"]:  # Dit zet (de meeste) missende categorieen naar de meest voorkomende
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
    for val in var_dict["list_vars"]:
        df[val] = df[val].map(lambda x: x if isinstance(x, list) else [])
    for var in var_dict["occurence_counts"]:
        df[var] = df[var].fillna(0)
        df[var] = df[var].map(lambda x: len(x) if isinstance(x, list) else x)

    for var in var_dict["bool_datetimes"]:
        na_cols = df[var].isna()
        na_cols = na_cols.map(lambda x: 1 if x == False else 0)
        df[var] = na_cols
    return df

def map_categories(df, category, cat_dict):
    df[category] = df[category].map(cat_dict)



def other_morph(frame, col_var, thresh, other_dict, is_list=False):
    if is_list:
        url_dict = {}
        for entry in frame[col_var]:
            if isinstance(entry, list):
                for url in entry:
                    if url in url_dict.keys():
                        url_dict[url] += 1
                    else:
                        url_dict[url] = 1
        good_vars = [key for key, value in sorted(url_dict.items(), key=lambda item: item[1], reverse=True)[:thresh]] # de urls die minder vaak dan thresh voorkomen
    else:
        good_vars = frame[col_var].value_counts().index[:thresh]

    if is_list:
        frame[col_var] = frame[col_var].map(lambda x: list(map(lambda y: "Other" if y not in good_vars else y, x)) if isinstance(x, list) else x)
    else:
        frame[col_var] = frame[col_var].map(lambda y: "Other" if y not in good_vars else y)

    other_dict[col_var] = good_vars
    return [frame, other_dict]





####################################################################################################
#### Augmentations
##################################################################################################
def augment_variables(frame, var_dict, url_dict, url_dict2, gitlink):
    augmented_vars = []
    orig_indices = frame.index
    frame["Visit_part_of_day"] = time_of_day(frame["visit_time"])
    augmented_vars.append("visit_time")
    var_dict["categorical_vars_median"].append("Visit_part_of_day")

    frame["has_partner"] = pd.to_numeric(has_partner(frame))
    var_dict["numeric_vars_mean_fill"].append("has_partner")
    augmented_vars += ["age", "age_partner"]
    frame["max_income"], frame["total_income"] = pd.to_numeric(income_augments(frame)[0]), pd.to_numeric(income_augments(frame)[1])
    var_dict["numeric_vars_mean_fill"].append("max_income")
    var_dict["numeric_vars_mean_fill"].append("total_income")
    augmented_vars += ["annual_income", "annual_income_partner"]
    frame, var_dict = morph_zip(frame, var_dict, gitlink)
    augmented_vars.append("mr_geo_zipcode")
    frame, var_dict = morph_city(frame, var_dict, gitlink)
    augmented_vars.append("mr_geo_city_name")
    frame, var_dict = search_keywords(frame, var_dict)
    augmented_vars.append("keywords")

    frame, var_dict = time_between_visits(frame, var_dict)
    augmented_vars.append("visit_timestamps")

    frame, var_dict = url_count(frame, var_dict)
    frame, var_dict = url_keywords(frame, var_dict)
    augmented_vars.append("url-name")
    subframe, var_dict = transform_urls(frame["recentlyvieweditems"], var_dict, url_dict, "url_mr")
    subframe.index = frame.index
    frame = frame.join(subframe)
    subframe, var_dict = transform_urls(frame["url-name"], var_dict, url_dict2, "url_heading")
    subframe.index = frame.index
    frame = frame.join(subframe)
    frame.index=orig_indices
    return [frame, var_dict, augmented_vars]



def morph_zip(df, var_dict, gitlink = ''):
    if gitlink!='':
        zip_df = pd.read_csv(gitlink + "/zipcode_final.csv")
    else:
        zip_df = pd.read_csv("zipcode_final.csv")
    zip_df["Postcode"] = zip_df["Postcode"].astype(int)
    df["zip_temp"] = df["mr_geo_zipcode"].map(lambda x: x if not isinstance(x, list) else x[0] if len(x) else None)
    zip_df["Total_income"] = zip_df["Bevolking"] * zip_df["Inkomen"]

    avg_income_person = zip_df["Total_income"].sum()/zip_df["Bevolking"].sum()
    avg_pop_zip = zip_df["Bevolking"].sum()/len(zip_df["Bevolking"])
    c=0
    income_population_segment = []
    while c<len(zip_df["Inkomen"]):
        if zip_df["Inkomen"][c]<=avg_income_person:
            if zip_df["Bevolking"][c]<=avg_pop_zip:
                income_population_segment.append("lowinc_lowpop") #laag inkomen laag bevolking
            else:
                income_population_segment.append("lowinc_highpop") #laag inkomen hoog bevolking
        else:
            if zip_df["Bevolking"][c]<=avg_pop_zip:
                income_population_segment.append("highinc_lowpop")  # laag inkomen laag bevolking
            else:
                income_population_segment.append("highinc_highpop")  # laag inkomen laag bevolking
        c+=1
    zip_df["income_population_segment"] = income_population_segment
    zip_df.columns = ["zip_code " + x for x in zip_df.columns]  # niet vergeten
    for column in zip_df.columns:
        if column not in ["zip_code Postcode", "zip_code income_population_segment"]: #alle numeric waardes
            var_dict["numeric_vars_mean_fill"].append(column)
            zip_df[column] = pd.to_numeric(zip_df[column])
        elif column!= "zip_code Postcode":
            var_dict["categorical_vars_median"].append(column)
    df["zip_temp"] = pd.to_numeric(df["zip_temp"], errors='coerce')
    merged = df.merge(zip_df, left_on="zip_temp", right_on="zip_code Postcode", how="left")  # zip_code zip_code as it has been transformed
    merged.index = df.index
    df = merged.drop(["zip_code Postcode", "zip_temp"], axis=1)
    return [df, var_dict]

def morph_city(frame, var_dict, gitlink = ''):
    if gitlink!='':
        gemeente_steden = pd.read_csv(gitlink + "/Gemeente_per_woonplaats.csv", delimiter=";", decimal=",")
    else:
        gemeente_steden = pd.read_csv("Gemeente_per_woonplaats.csv", delimiter=";", decimal=",")
    gemeente_stad_dict = dict(zip(gemeente_steden["Stad"], gemeente_steden["Gemeente"]))
    steden = frame["mr_geo_city_name"].map(lambda x: x if not isinstance(x, list) else x[0] if len(x) else None)
    gemeentes = pd.DataFrame({"Gemeente": steden.map(gemeente_stad_dict)})

    if gitlink != '':
        gemeente_stats = pd.read_csv(gitlink + "/Kerncijfers_per_gemeente.csv",delimiter=";", decimal=",")
    else:
        gemeente_stats = pd.read_csv("Kerncijfers_per_gemeente.csv", delimiter=";", decimal=",")
    gemeente_stats.columns = ["Gemeente :" + col for col in gemeente_stats.columns]
    merged = gemeentes.merge(gemeente_stats, left_on="Gemeente", right_on="Gemeente :Gemeente", how="left")
    merged= merged.drop(['Gemeente', "Gemeente :Gemeente"], axis=1)

    merged["Gemeente :Inkomen"] = merged["Gemeente :Inkomen"].map(lambda x: x.replace(',', '.') if isinstance(x, str) and re.search(r".+,.+",x) else None)
    merged["Gemeente :Inkomen"] = pd.to_numeric(merged["Gemeente :Inkomen"])
    var_dict["numeric_vars_mean_fill"] += list(merged.columns)
    merged.index=frame.index
    conc_frame = frame.join(merged)
    return [conc_frame, var_dict]

import datetime

def is_weekend(df, date_vars, var_dict):
    for date_var in date_vars:
        datetime_series = df[date_var].fillna(df[date_var].median())
        datetime_series = datetime_series.map(lambda x: datetime.datetime.fromtimestamp(x/1e3) if abs(x)<1e13 else datetime.datetime.fromtimestamp(x/1e9))
        is_weekend_series = datetime_series.map(lambda x: 1 if x.weekday()>4 else 0)
        df[date_var + ' is_weekend'] = is_weekend_series
        var_dict["numeric_vars_mean_fill"].append(date_var + ' is_weekend')
    return [df, var_dict]

import regex as re
def time_of_day(visit_time):
    vals = []
    for list_value in visit_time:
        try:
            value = list_value[0]
            if re.search(r"[5678]\sAM -" , value):
                vals.append("Early Morning")
            elif re.search(r"(9|10|11)\sAM -" , value):
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
        except: #nan
            vals.append(None)
    return vals

def time_between_visits(frame, var_dict):
    avg_time = frame["visit_timestamps"].map(lambda x: (int(x[0])-int(x[len(x)-1])/len(x) if type(x)== list and len(x)>1 else np.nan)) #eerste - laatste/aantal is het gemiddelde
    visit_amt = frame["visit_timestamps"].map(lambda x: len(x) if type(x)==list else 0)
    mr_time_between = frame["visit_timestamps"].map(lambda x: int(x[len(x)-1])-int(x[len(x)-2]) if type(x)==list and len(x)>1 else np.nan)
    frame["avg_time_between_visits"], frame["visit_amt"], frame["mr_time_between_visits"] = avg_time, visit_amt, mr_time_between
    var_dict["categorical_vars_none"].remove("visit_timestamps")
    frame = frame.drop("visit_timestamps", axis=1)
    var_dict["numeric_vars_mean_fill"]+= ["avg_time_between_visits", "mr_time_between_visits"]
    var_dict["numeric_vars_zero_fill"] += ["visit_amt"]
    return [frame, var_dict]

def has_partner(frame):
    nans = frame["age_partner"].notna()
    bools = nans.map(lambda x: 1 if x==True else 0)
    return bools

def search_keywords(frame, var_dict): #doet niks, niemand zoekt blijkbaar
    user_searches = frame["keywords"]
    bestaande_woning_keywords = ["oversluiten", "verbouw", 'tweede', "extra", "rentevaste periode",
                                 "pensioen", "rentemiddeling"]
    starter_keywords = ["studie", "eerste"]
    volgende_woning_keywords = ["meenemen", "overwaarde", "restschuld", "groter", "verhuizing", "verkoop",
                                "tweede", "overbrug", "2e"]
    bestaande_woning_searches, starter_searches, volgende_woning_searches = np.zeros(len(user_searches)),np.zeros(len(user_searches)),np.zeros(len(user_searches))
    c=0
    for searches in user_searches: #one can have multiple searches
        if isinstance(searches, list):
            for search in searches:
                for keyword in starter_keywords:
                    if keyword in search:
                        starter_searches[c]+=1
                for keyword in bestaande_woning_keywords:
                    if keyword in search:
                        bestaande_woning_searches[c]+=1
                for keyword in volgende_woning_keywords:
                    if keyword in search:
                        volgende_woning_searches[c] += 1
        c+=1
    frame["search bestaande_woning"], frame["search volgende_woning"], frame["search starter"] = bestaande_woning_searches, volgende_woning_searches, starter_searches
    var_dict["numeric_vars_zero_fill"] += ["search bestaande_woning", "search volgende_woning", "search starter"]
    return [frame, var_dict]

def url_count(frame, var_dict):
    frame["total_url_counts"] = frame["url-name"].map(lambda x: len(x) if isinstance(x, list) else 0)
    visits = frame["visits"].map(lambda x: int(x) if not np.isnan(x) and not x==0 else 1)
    frame["avg_url_counts"] = frame["total_url_counts"]/visits
    var_dict["numeric_vars_zero_fill"]+= ["total_url_counts", "avg_url_counts"]
    return frame, var_dict

def url_keywords(frame, var_dict): #url has been formatted already
    user_url_headings = frame["url-name"]
    mr_urls = frame["recentlyvieweditems"]
    bestaande_woning_keywords = ["oversluiten", "verbouw", 'tweede', "extra", "rentevaste periode",
                                 "pensioen", "rentemiddeling", "uit elkaar", "scheiden", "hypotheek aanpassen", "tweede hypotheek",
                                 "hypotheek-aanpassen", "tweede-hypotheek", "uit-elkaar", "rentevaste-periode"]
    starter_keywords = ["studie", "eerste woning", "eerste huis", "mezelf wonen", "starter",
                        "alles wat je moet weten als je een huis wilt kopen",
                        "eerste-woning", "eerste-huis", "mezelf-wonen"]
    volgende_woning_keywords = ["volgende woning","volgende huis", "overwaarde", "tweede woning", "woning waard",
                                "volgende-woning","volgende-huis", "overwaarde", "tweede-woning", "woning-waard"]

    bestaande_woning_urls, starter_urls, volgende_woning_urls = np.zeros(len(user_url_headings)), np.zeros(len(user_url_headings)), np.zeros(len(user_url_headings))

    for c in range(0, len(user_url_headings)): #one can have multiple searches
        urls_headings = user_url_headings.iloc[c]
        indiv_urls = mr_urls.iloc[c]
        if isinstance(urls_headings, list):
            for url in urls_headings:
                for keyword in starter_keywords:
                    if keyword in url:
                        starter_urls[c]+=1
                for keyword in bestaande_woning_keywords:
                    if keyword in url:
                        bestaande_woning_urls[c]+=1
                for keyword in volgende_woning_keywords:
                    if keyword in url:
                        volgende_woning_urls[c] += 1
        if isinstance(indiv_urls, list):
            for url in indiv_urls:
                for keyword in starter_keywords:
                    if keyword in url:
                        starter_urls[c]+=1
                for keyword in bestaande_woning_keywords:
                    if keyword in url:
                        bestaande_woning_urls[c]+=1
                for keyword in volgende_woning_keywords:
                    if keyword in url:
                        volgende_woning_urls[c] += 1

    frame["url_keywords bestaande_woning"], frame["url_keywords volgende_woning"], frame["url_keywords starter"] = bestaande_woning_urls, volgende_woning_urls, starter_urls
    var_dict["numeric_vars_zero_fill"] += ["url_keywords bestaande_woning", "url_keywords volgende_woning", "url_keywords starter"]
    return [frame, var_dict]

def income_augments(frame):
    income = frame["annual_income"]
    partner_income = frame["annual_income_partner"]
    max_income = []
    total_income = []
    c=0
    while c<len(income):
        max_income.append(max(income.iloc[c], partner_income.iloc[c]))
        total_income.append(income.iloc[c] + partner_income.iloc[c])
        c+=1
    return[max_income, total_income]



#####
#Clusters

def transform_urls(urls_list, var_dict, url_dict, name):
    mapped_list = []
    for url_list in urls_list:
        mapped = []
        if isinstance(url_list, list): #catches Nan
            for url in url_list:
                if url in url_dict.keys():
                    mapped.append(url_dict[url])
        mapped_list.append(mapped)

    df = pd.DataFrame(0, index = np.arange(len(urls_list)), columns=set(url_dict.values()))
    df.columns = [name + " " + str(x) for x in df.columns]
    for x in range(0, len(mapped_list)):
        value_counts = pd.Series(mapped_list[x]).value_counts()
        for index in value_counts.index:
            row = df.loc[x]
            row[index] = value_counts[index]
    var_dict["numeric_vars_zero_fill"] += list(df.columns)

    return [df, var_dict]


#####
#Conversie
from imblearn.over_sampling import SMOTE
def smote_fix_db(x, class_label, wanted_proportion):
    y = x[class_label]
    sm = SMOTE(sampling_strategy=wanted_proportion/(1-wanted_proportion))
    x, y = sm.fit_resample(x, y)
    x= pd.DataFrame(x)
    x[class_label] = y
    return x







class MultiLabelCounter():
    def __init__(self, classes=None):
        self.classes_ = classes

    def fit(self,y):
        self.classes_ = sorted(set(itertools.chain.from_iterable(y)))
        self.mapping = dict(zip(self.classes_,
                                         range(len(self.classes_))))
        return self

    def transform(self,y):
        yt = []
        for labels in y:
            data = [0]*len(self.classes_)
            for label in labels:
                data[self.mapping[label]] +=1
            yt.append(data)
        return yt

    def fit_transform(self,y):
        return self.fit(y).transform(y)

class TargetEncoder():
    def __init__(self, classes=None):
        self.classes_ = classes
    def fit(self,series, target):
        self.classes_ = sorted(set(itertools.chain.from_iterable(series)))
        df = pd.DataFrame(zip(series, target), columns=["x", "y"])
        target_classes = sorted(target.unique())
        if target_classes>2:
            for cl in self.classes:
                co_ocs = []
                for cl_y in target_classes:
                    co_oc = len(df[df["x"] == cl and df["y"] == cl_y]) / len(df[df["x"] == cl])
                    co_ocs.append(co_oc)
        else:
            for cl in self.classes:
                co_oc = len(df[df["x"]==cl and df["y"]==1])/len(df[df["x"]==cl])
        self.mapping = dict(zip(self.classes_,
                                         range(len(self.classes_))))
        return self

    def transform(self,y):
        yt = []
        for labels in y:
            data = [0]*len(self.classes_)
            for label in labels:
                data[self.mapping[label]] +=1
            yt.append(data)
        return yt

    def fit_transform(self,y):
        return self.fit(y).transform(y)
