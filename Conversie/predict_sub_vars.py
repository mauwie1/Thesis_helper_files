from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cf_mat
from sklearn.ensemble import RandomForestClassifier as rfc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score


def regress_fill(var, not_considered, raw_frame, model, train_x, validation_x, test_x, scale = False):
    raw_age = raw_frame.dropna(subset=[var])
    missing_indices = set(raw_frame.index) - set(raw_age.index)
    print(len(missing_indices))
    y = train_x[var].loc[train_x.index & raw_age.index]
    x = train_x.drop(not_considered + [var], axis=1).loc[train_x.index & raw_age.index]
    if scale:
        x = normalize(x)
    model.fit(x, y)
    print("dummy R2 :" + str(model.score(validation_x.drop(not_considered + [var], axis=1).loc[validation_x.index & raw_age.index], validation_x[var].loc[validation_x.index & raw_age.index])))
    upd_train_x, upd_validation_x, upd_test_x = train_x.copy(deep=True), validation_x.copy(deep=True), test_x.copy(deep=True)
    for xje in [upd_train_x, upd_validation_x, upd_test_x]:
        missing = missing_indices & set(xje.index)
        missing_df = xje.drop(not_considered + [var], axis=1).loc[missing]
        preds = model.predict(missing_df)
        zip_series = pd.Series(preds, index=missing_df.index)
        print("amount filled : " + str(len(zip_series)))
        xje[var].update(zip_series)
    importances = sorted(list(zip(train_x.columns, model.feature_importances_)), key=lambda z: z[1], reverse=True)
    return [preds, model, importances, upd_train_x, upd_validation_x, upd_test_x]

def predict_fill(var, not_considered, raw_frame, model, train_x, validation_x, test_x, scale = False):
    raw_age = raw_frame.dropna(subset=[var])
    missing_indices = set(raw_frame.index) - set(raw_age.index)
    y = train_x[var].loc[train_x.index & raw_age.index]
    x = train_x.drop(not_considered + [var], axis=1).loc[train_x.index & raw_age.index]
    if scale:
        x = normalize(x)
    model.fit(x, y)
    preds = model.predict(validation_x.drop(not_considered + [var], axis=1).loc[validation_x.index & raw_age.index])
    print("dummy F1 :" + str(round(f1_score(preds, validation_x[var].loc[validation_x.index & raw_age.index]), 4)))
    upd_train_x, upd_validation_x, upd_test_x = train_x.copy(deep=True), validation_x.copy(deep=True), test_x.copy(deep=True)
    for xje in [upd_train_x, upd_validation_x, upd_test_x]:
        missing = missing_indices & set(xje.index)
        missing_df = xje.drop(not_considered + [var], axis=1).loc[missing]
        preds = model.predict(missing_df)
        zip_series = pd.Series(preds, index=missing_df.index)
        print("amount filled : " + str(len(zip_series)))
        xje[var].update(zip_series)
    #importances = sorted(list(zip(cols, model.feature_importances_)), key=lambda z: z[1], reverse=True)
    return [preds, model, upd_train_x, upd_validation_x, upd_test_x]

def regress_add(var, not_considered, raw_frame, model, train_x, validation_x, test_x):
    raw_age = raw_frame.dropna(subset=[var])
    missing_indices = set(raw_frame.index) - set(raw_age.index)

    concat_frame = pd.concat([train_x, validation_x, test_x])
    sub_frame = concat_frame.loc[raw_age.index]
    y = sub_frame[var]
    x = sub_frame.drop(not_considered + [var], axis=1)
    model.fit(x, y)
    print("dummy R2 :" + str(model.score(x, y)))

    preds = model.predict(concat_frame.drop(not_considered + [var], axis=1).loc[missing_indices])

    upd_train_x, upd_validation_x, upd_test_x = train_x.copy(deep=True), validation_x.copy(deep=True), test_x.copy(deep=True)
    for xje in [upd_train_x, upd_validation_x, upd_test_x]:
        preds = model.predict(xje.drop(not_considered + [var], axis=1))
        xje["pred" + var] = preds

    importances = sorted(list(zip(x.columns, model.feature_importances_)), key=lambda z: z[1], reverse=True)
    return [preds, model, importances, upd_train_x, upd_validation_x, upd_test_x]

def fill_zero(var, raw_frame, train_x, validation_x, test_x):
    raw_age = raw_frame.dropna(subset=[var])
    missing_indices = set(raw_frame.index) - set(raw_age.index)
    upd_train_x, upd_validation_x, upd_test_x = train_x.copy(deep=True), validation_x.copy(deep=True), test_x.copy(deep=True)
    for xje in [upd_train_x, upd_validation_x, upd_test_x]:
        missing = missing_indices & set(xje.index)

        zip_series = pd.Series(0, index=missing)
        print("amount filled : " + str(len(zip_series)))
        xje[var].update(zip_series)
    return [upd_train_x, upd_validation_x, upd_test_x]



def predict_category(var, not_considered, raw_frame, model, train_x, validation_x, test_x, columns='', smote=-1, scale=False):
    sub_vars = []
    for col in train_x.columns:
        if col.split()[0] == var:
            sub_vars.append(col)

    not_na_var = raw_frame.dropna(subset=[var]).index
    missing_indices = set(raw_frame.index) - set(not_na_var)
    print("Missing train :" + str(len(missing_indices & set(train_x.index))))

    frames = []


    for x in [train_x, validation_x, test_x]:
        var_frame = x[sub_vars]
        cats = [" ".join(x.split()[1:]) for x in var_frame.columns]
        var_frame.columns = cats
        if len(sub_vars) == 1:
            cat_frame = var_frame[cats[0]].map(lambda x: cats[0] if x ==1 else "Other") #arbitrair, valt toch weg
        else:
            cat_frame = var_frame.idxmax(axis=1)
        var_y = cat_frame.loc[not_na_var & cat_frame.index]
        var_x = x.loc[var_y.index & x.index]
        if isinstance(columns, list):
            var_x = var_x.drop(not_considered + sub_vars, axis=1)
            var_x = var_x[set(var_x.columns)&set(columns)]
        else:
            var_x = var_x.drop(not_considered + sub_vars, axis=1)
        frames.append([var_x, var_y])

    tr_x, tr_y = frames[0]
    if smote!=-1:
        smote_fix(tr_x, tr_y, smote)

    model = model.fit(tr_x, tr_y)
    val_x, val_y = frames[1]
    print(model.predict(val_x))
    preds = model.predict(val_x)
    print("dummy F1 :" + str(f1_score(preds, val_y, average="macro")))
    print(cf_mat(val_y, model.predict(val_x)))
    upd_train_x, upd_validation_x, upd_test_x = train_x.copy(deep=True), validation_x.copy(deep=True), test_x.copy(deep=True)
    #for col in dummy_preds.columns:
    #       dummy_preds[col].values[:] = 2#?? weghalen of iets ander verzinnen
    c=0
    for xje in [upd_train_x, upd_validation_x, upd_test_x]:
        indices = set(xje.index) & missing_indices

        if isinstance(columns, list):
            preds = model.predict(xje[tr_x.columns].loc[indices])
            print(preds)
        else:
            preds = model.predict(xje.loc[indices].drop(not_considered + sub_vars, axis=1))
        dummy_preds = pd.get_dummies(pd.Series(preds, index=indices))
        dummy_preds.columns = [var + " " + col for col in dummy_preds.columns]
        xje.update(dummy_preds.loc[indices])
        xje[sub_vars] = xje[sub_vars].astype(int)
        c+=1



    importances = sorted(list(zip(x.columns, model.feature_importances_)), key=lambda z: z[1], reverse=True)
    return [preds, model, importances, upd_train_x, upd_validation_x, upd_test_x]


def smote_fix(x, y, wanted_proportion):
    if max(y.value_counts())/len(y) < wanted_proportion:
        sm = SMOTE(sampling_strategy=wanted_proportion/(1-wanted_proportion))
        x_res, y_res = sm.fit_resample(x, y)
        return [x_res, y_res]
    return(x, y)