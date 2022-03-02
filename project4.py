from collections import Counter
from itertools import combinations
from math import sqrt
import random
from keras.layers import Concatenate, Dense, Dot, Dropout, Embedding, Input, Reshape
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow

STUDENT_ID = '20622268'


# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


def build_cfmodel(n_users, n_items, embed_size, output_layer='dot'):
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    
    user_emb = Embedding(output_dim=embed_size, input_dim=n_users, input_length=1)(user_input)
    user_emb = Reshape((embed_size,))(user_emb)
    item_emb = Embedding(output_dim=embed_size, input_dim=n_items, input_length=1)(item_input)
    item_emb = Reshape((embed_size,))(item_emb)
    
    if output_layer == 'dot':
        model_output = Dot(axes=1)([user_emb, item_emb])
    elif output_layer == 'mlp':
        mlp_input = Concatenate()([user_emb, item_emb])

        dense_1 = Dense(64, activation='relu')(mlp_input)
        dense_1_dp = Dropout(0.15)(dense_1)
        dense_2 = Dense(32, activation='relu')(dense_1_dp)
        dense_2_dp = Dropout(0.15)(dense_2)
        model_output = Dense(1)(dense_2_dp)
    else:
        raise NotImplementedError

    model = Model(inputs=[user_input, item_input],
                  outputs=model_output)
    return model
 
def build_deepwide_model(len_continuous, deep_vocab_lens, deep2_vocab_lens, len_wide, embed_size):
    embed_size2 = 32
    
    input_list = []
    continuous_input = Input(shape=(len_continuous,), dtype='float32', name='continuous_input')
    input_list.append(continuous_input)

    emb_list = []
    for vocab_size in deep_vocab_lens:
        embed_size = int(vocab_size**0.25)
        _input = Input(shape=(1,), dtype='int32')
        input_list.append(_input)
        _emb = Embedding(output_dim=embed_size, input_dim=vocab_size, input_length=1)(_input)
        _emb = Reshape((embed_size,))(_emb)
        emb_list.append(_emb)
    
    for vocab_size in deep2_vocab_lens:
        embed_size2 = int(vocab_size**0.25)
        _input = Input(shape=(1,), dtype='int32')
        input_list.append(_input)
        _emb2 = Embedding(output_dim=embed_size2, input_dim=vocab_size, input_length=1)(_input)
        _emb2 = Reshape((embed_size2,))(_emb2)
        emb_list.append(_emb2)
        
    deep_input = Concatenate()(emb_list + [continuous_input])
    dense_1 = Dense(512, activation='relu')(deep_input)
    dense_1_dp = Dropout(0.3)(dense_1)
    dense_2 = Dense(256, activation='relu')(dense_1_dp)
    dense_2_dp = Dropout(0.3)(dense_2)
    dense_3 = Dense(128, activation='relu')(dense_2_dp)
    dense_3_dp = Dropout(0.3)(dense_3)

    wide_input = Input(shape=(len_wide,), dtype='float32')
    input_list.append(wide_input)
    
    fc_input = Concatenate()([dense_3_dp, wide_input])
    dense_1 = Dense(8, activation='sigmoid')(fc_input)
    model_output = Dense(1)(dense_1)
    model = Model(inputs=input_list,
                  outputs=model_output)
    return model


def get_continuous_features(df, continuous_columns):
    continuous_features = df[continuous_columns].values
#     continuous_features = df[continuous_columns].a
    return continuous_features


def get_top_k_p_combinations(df, comb_p, topk, output_freq=False):
    def get_category_combinations(categories_str, comb_p=2):
        categories = categories_str.split(', ')
        return list(combinations(categories, comb_p))
    all_categories_p_combos = df["item_categories"].apply(
        lambda x: get_category_combinations(x, comb_p)).values.tolist()
    all_categories_p_combos = [tuple(t) for item in all_categories_p_combos for t in item]
    tmp = dict(Counter(all_categories_p_combos))
    sorted_categories_combinations = list(sorted(tmp.items(), key=lambda x: x[1], reverse=True))
    if output_freq:
        return sorted_categories_combinations[:topk]
    else:
        return [t[0] for t in sorted_categories_combinations[:topk]]


def get_wide_features(df):
    def categories_to_binary_output(categories):
        binary_output = [0 for _ in range(len(selected_categories_to_idx))]
        for category in categories.split(', '):
            if category in selected_categories_to_idx:
                binary_output[selected_categories_to_idx[category]] = 1
            else:
                binary_output[0] = 1
        return binary_output
    def categories_cross_transformation(categories):
        current_category_set = set(categories.split(', '))
        corss_transform_output = [0 for _ in range(len(top_combinations))]
        for k, comb_k in enumerate(top_combinations):
            if len(current_category_set & comb_k) == len(comb_k):
                corss_transform_output[k] = 1
            else:
                corss_transform_output[k] = 0
        return corss_transform_output

    category_binary_features = np.array(df.item_categories.apply(
        lambda x: categories_to_binary_output(x)).values.tolist())
    category_corss_transform_features = np.array(df.item_categories.apply(
        lambda x: categories_cross_transformation(x)).values.tolist())
    return np.concatenate((category_binary_features, category_corss_transform_features), axis=1)


root_path = ""
tr_df = pd.read_csv(root_path + "data/train.csv")
val_df = pd.read_csv(root_path + "data/valid.csv")
te_df = pd.read_csv(root_path + "data/test.csv")

tr_ratings = tr_df.stars.values
val_ratings = val_df.stars.values

user_df = pd.read_json(root_path + "data/user.json")
item_df = pd.read_json(root_path + "data/business.json")
user_df = user_df.rename(index=str, columns={t: 'user_' + t for t in user_df.columns if t != 'user_id'})
item_df = item_df.rename(index=str, columns={t: 'item_' + t for t in item_df.columns if t != 'business_id'})

tr_df["index"] = tr_df.index
val_df["index"]  = val_df.index
te_df["index"] = te_df.index
tr_df = pd.merge(pd.merge(tr_df, user_df, on='user_id'), item_df, on='business_id').sort_values(by=['index']).reset_index(drop=True)
val_df = pd.merge(pd.merge(val_df, user_df, on='user_id'), item_df, on='business_id').sort_values(by=['index']).reset_index(drop=True)
te_df = pd.merge(pd.merge(te_df, user_df, on='user_id'), item_df, on='business_id').sort_values(by=['index']).reset_index(drop=True)

print("Prepare continuous features...")
continuous_columns = ["user_average_stars", "user_cool", "user_fans", 
                      "user_review_count", "user_useful", "user_funny",
                      "item_is_open", "item_latitude", "item_longitude", 
                      "item_review_count", "item_stars",
                      "user_compliment_cool", "user_compliment_cute", "user_compliment_funny",
                      "user_compliment_hot"] #VALID RMSE:  1.0429724102351896
#                           "user_compliment_list", "user_compliment_more",
#                           "user_compliment_note", "user_compliment_photos", "user_compliment_plain",
#                           "user_compliment_profile", "user_compliment_writer"]
tr_continuous_features = get_continuous_features(tr_df, continuous_columns)
val_continuous_features = get_continuous_features(val_df, continuous_columns)
te_continuous_features = get_continuous_features(te_df, continuous_columns)
scaler = StandardScaler().fit(tr_continuous_features)
tr_continuous_features = scaler.transform(tr_continuous_features)
val_continuous_features = scaler.transform(val_continuous_features)
te_continuous_features = scaler.transform(te_continuous_features)

import re
pattern = re.compile("'(.*)'")

item_df['item_attributes_Alcohol'] = item_df['item_attributes'].apply(lambda x: str(x['Alcohol']) if x != None and 'Alcohol' in x else '\'none\'')
item_df['item_attributes_Alcohol'] = item_df['item_attributes_Alcohol'].apply(lambda x: pattern.findall(x)[0] if len(pattern.findall(x)) > 0 else x)
item_df['item_attributes_Alcohol'] = item_df['item_attributes_Alcohol'].apply(lambda x: x.lower())
item_df['item_attributes_WiFi'] = item_df['item_attributes'].apply(lambda x: str(x['WiFi']) if x != None and 'WiFi' in x else '\'none\'')
item_df['item_attributes_WiFi'] = item_df['item_attributes_WiFi'].apply(lambda x: pattern.findall(x)[0] if len(pattern.findall(x)) > 0 else x)
item_df['item_attributes_WiFi'] = item_df['item_attributes_WiFi'].apply(lambda x: x.lower())
item_df['item_attributes_NoiseLevel'] = item_df['item_attributes'].apply(lambda x: str(x['NoiseLevel']) if x != None and 'NoiseLevel' in x else '\'none\'')
item_df['item_attributes_NoiseLevel'] = item_df['item_attributes_NoiseLevel'].apply(lambda x: pattern.findall(x)[0] if len(pattern.findall(x)) > 0 else x)
item_df['item_attributes_NoiseLevel'] = item_df['item_attributes_NoiseLevel'].apply(lambda x: x.lower())
item_df['item_attributes_HasTV'] = item_df['item_attributes'].apply(lambda x: str(x['HasTV']) if x != None and 'HasTV' in x else '\'none\'')
item_df['item_attributes_HasTV'] = item_df['item_attributes_HasTV'].apply(lambda x: pattern.findall(x)[0] if len(pattern.findall(x)) > 0 else x)
item_df['item_attributes_HasTV'] = item_df['item_attributes_HasTV'].apply(lambda x: x.lower())

print("Prepare deep features...")
item_deep_columns = ["item_city", "item_postal_code", "item_state"]
item_deep_vocab_lens = []
for col_name in item_deep_columns:
    # transpose Category features to number
    tmp = item_df[col_name].unique()
    vocab = dict(zip(tmp, range(1, len(tmp) + 1)))
    item_deep_vocab_lens.append(len(vocab) + 1)
    item_df[col_name + "_idx"] = item_df[col_name].apply(lambda x: vocab[x] if x in vocab else 0)
item_deep_idx_columns = [t + "_idx" for t in item_deep_columns]
item_to_deep_features = dict(zip(item_df.business_id.values, item_df[item_deep_idx_columns].values.tolist()))
tr_deep_features = np.array(tr_df.business_id.apply(lambda x: item_to_deep_features[x]).values.tolist())
val_deep_features = np.array(val_df.business_id.apply(lambda x: item_to_deep_features[x]).values.tolist())
te_deep_features = np.array(te_df.business_id.apply(lambda x: item_to_deep_features[x]).values.tolist())

print("Prepare deep2 features...")
item_deep2_columns = ["item_attributes_Alcohol", "item_attributes_WiFi",
                      "item_attributes_NoiseLevel", "item_attributes_HasTV"]
item_deep2_vocab_lens = []
for col_name in item_deep2_columns:
    # transpose Category features to number
    tmp = item_df[col_name].unique()
    vocab = dict(zip(tmp, range(1, len(tmp) + 1)))
    item_deep2_vocab_lens.append(len(vocab) + 1)
    item_df[col_name + "_idx"] = item_df[col_name].apply(lambda x: vocab[x] if x in vocab else 0)
item_deep2_idx_columns = [t + "_idx" for t in item_deep2_columns]
item_to_deep2_features = dict(zip(item_df.business_id.values, item_df[item_deep2_idx_columns].values.tolist()))
tr_deep2_features = np.array(tr_df.business_id.apply(lambda x: item_to_deep2_features[x]).values.tolist())
val_deep2_features = np.array(val_df.business_id.apply(lambda x: item_to_deep2_features[x]).values.tolist())
te_deep2_features = np.array(te_df.business_id.apply(lambda x: item_to_deep2_features[x]).values.tolist())

# Wide (Category) features
print("Prepare wide features...")
#   Prepare binary encoding for each selected categories
all_categories = [category for category_list in item_df.item_categories.values for category in category_list.split(", ")]
category_sorted = sorted(Counter(all_categories).items(), key=lambda x: x[1], reverse=True)
selected_categories = [t[0] for t in category_sorted[:500]]
selected_categories_to_idx = dict(zip(selected_categories, range(1, len(selected_categories) + 1)))
selected_categories_to_idx['unk'] = 0
idx_to_selected_categories = {val: key for key, val in selected_categories_to_idx.items()}
#   Prepare Cross transformation for each categories
top_combinations = []
top_combinations += get_top_k_p_combinations(tr_df, 2, 50, output_freq=False)
top_combinations += get_top_k_p_combinations(tr_df, 3, 30, output_freq=False)
top_combinations += get_top_k_p_combinations(tr_df, 4, 20, output_freq=False)
top_combinations = [set(t) for t in top_combinations]

tr_wide_features = get_wide_features(tr_df)
val_wide_features = get_wide_features(val_df)
te_wide_features = get_wide_features(te_df)

tr_features = []
tr_features.append(tr_continuous_features.tolist())
tr_features += [tr_deep_features[:,i].tolist() for i in range(len(tr_deep_features[0]))]
tr_features += [tr_deep2_features[:,i].tolist() for i in range(len(tr_deep2_features[0]))]
tr_features.append(tr_wide_features.tolist()) # shape(5, 100000)
val_features = []
val_features.append(val_continuous_features.tolist())
val_features += [val_deep_features[:,i].tolist() for i in range(len(val_deep_features[0]))]
val_features += [val_deep2_features[:,i].tolist() for i in range(len(val_deep2_features[0]))]
val_features.append(val_wide_features.tolist())
te_features = []
te_features.append(te_continuous_features.tolist())
te_features += [te_deep_features[:,i].tolist() for i in range(len(te_deep_features[0]))]
te_features += [te_deep2_features[:,i].tolist() for i in range(len(te_deep2_features[0]))]
te_features.append(te_wide_features.tolist())


# Model training
deepwide_model = build_deepwide_model(
    len(tr_continuous_features[0]),
    item_deep_vocab_lens, 
    item_deep2_vocab_lens,
    len(tr_wide_features[0]),
    embed_size=128)
deepwide_model.compile(optimizer='adagrad', loss='mse')

filepath=root_path + "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#     checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#     print(type(tr_features))
history = deepwide_model.fit(
    tr_features,
    tr_ratings,
#         validation_split=0.2,
    epochs=5, verbose=1, callbacks=callbacks_list)

#     deepwide_model.load_weights(root_path + "weights.best.hdf5")

# Make Prediction
y_pred = deepwide_model.predict(tr_features)
print("TRAIN RMSE: ", rmse(y_pred, tr_ratings))
y_pred = deepwide_model.predict(val_features)
print("VALID RMSE: ", rmse(y_pred, val_ratings))
y_pred = deepwide_model.predict(te_features)
res_df = pd.DataFrame()
res_df['pred'] = y_pred[:, 0]
res_df.to_csv("{}.csv".format(STUDENT_ID), index=False)
print("Writing test predictions to file done.")