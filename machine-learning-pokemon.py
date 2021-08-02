# -*- coding: utf-8 -*-
"""
Machine-learning script made from here: 

    https://www.realpythonproject.com/build-a-streamlit-app-to-predict-if-a-pokemon-is-legendary/

Uses scikit-learn for machine-learning and streamlit to make it into an app.
"""

# Pokémon Machine learning project.

## Imports
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score
)

LEGEND_LABELS = ["Normal", "Legendary"]

## Helper functions
def title(s):
    """Helper function to display a title between two blank lines."""
    st.text("")
    st.title(s)
    st.text("")

def clean_and_split(df):
    """
    Clean and format the data in the DataFrame for the machine-learning script.
    """
    legendary_df = df[df.legendary]
    normal_df = df[df.legendary==False].sample(75)
    legendary_df.fillna(legendary_df.mean(), inplace=True)
    normal_df.fillna(normal_df.mean(), inplace=True)
    feature_list = ["weight_kg", "height_m", "hp", "attack", "defense",
                    "sp_attack", "sp_defense", "speed", "base_total", "legendary"]
    sub_df = pd.concat([legendary_df, normal_df])[feature_list]
    X = sub_df.loc[:, sub_df.columns != "legendary"]
    y = sub_df["legendary"]
    X_train, X_test, y_train, y_test, = train_test_split(
        X, y, random_state=1, test_size=0.2, shuffle=True, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

## Intro
st.title("Is that a Legendary Pokémon?")
st.image("079Slowpoke-Galar.png")
# st.components.v1.html("""
# <img src="079Slowpoke-Galar.png" width=100%>
# """)
st.markdown("""
Image from [Bulbapedia](https://bulbapedia.bulbagarden.net/wiki/File:079Slowpoke-Galar.png).
""")

## Load data
df = pd.read_csv("pokemon_db.csv", sep="|")
# df = df[df.gen <= 5]
st.dataframe(df.head())

## Basic Info
shape = df.shape
num_total = len(df)
num_legendary = len(df[df.legendary == True])
num_non_legendary = num_total - num_legendary

st.subheader(f"""
Number of Pokémon: {num_total}
""")

st.subheader(f"""
Number of Legendary Pokémon: {num_legendary}
""")

st.subheader(f"""
Number of Non-Legendary Pokémon: {num_non_legendary}
""")

st.subheader(f"""
Number of Features: {shape[1]}
""")

## Legendary Pokemon by type graph
title('Legendary Pokémon Distribution based on Type')
legendary_df = df[df['legendary'] == True]
normal_df = df[df.legendary == False]
fig1 = plt.figure()
ax = sns.countplot(data=legendary_df, x='type_1', order=(legendary_df['type_1']).value_counts().index)
plt.xticks(rotation=45)
st.pyplot(fig1)
fig1_2 = plt.figure()
ax = sns.countplot(data=legendary_df, x='type_2', order=(legendary_df['type_2']).value_counts().index)
plt.xticks(rotation=45)
st.pyplot(fig1_2)

## Height vs Weight
title('Height vs Weight for Legendary and Non-Legendary Pokémon')
fig2 = plt.figure()
ax2 = sns.scatterplot(data=df, x = 'weight_kg', y = 'height_m', hue='legendary')
st.pyplot(fig2)

## Correlation between features
title('Correlation between features \n(Legendary Pokémon only)')
fig3 = plt.figure()
ax3 = sns.heatmap(legendary_df[["hp",'attack','defense','sp_attack','sp_defense','speed', "base_total",'height_m','weight_kg']].corr())
ax3.axis("equal")
st.pyplot(fig3)

## Correlation between features of normal pokemon
title('Correlation between features \n(Non-legendary Pokémon, includes Mythical Pokémon)')
fig3_2 = plt.figure()
ax3_2 = sns.heatmap(normal_df[["hp",'attack','defense','sp_attack','sp_defense','speed', "base_total",'height_m','weight_kg']].corr())
ax3_2.axis("equal")
st.pyplot(fig3_2)

## Special Attack vs Attack
title("Special Attack vs Attack")
fig4 = plt.figure()
ax4 = sns.scatterplot(data=df, x='sp_attack', y='attack', hue='legendary')
ax4.axis("equal")
st.pyplot(fig4)

## Random Forest
title('Random Forest')
X_train, X_test, y_train, y_test = clean_and_split(df)
st.subheader("Sample Data")
st.dataframe(X_train.head(3))
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train , y_train)

## Metrics
title("Metrics")
st.subheader(f"Model Score: {model.score(X_test, y_test)}")
st.subheader(f"Precision Score: {precision_score(model.predict(X_test), y_test)}")
st.subheader(f"Recall Score: {recall_score(model.predict(X_test), y_test)}")

## Confusion Matrix
st.subheader("Confusion Matrix")
fig5 = plt.figure()
conf_matrix = confusion_matrix(model.predict(X_test), y_test)
ax5 = sns.heatmap(conf_matrix, annot=True, xticklabels=['Normal', 'Legendary'], yticklabels=['Normal', 'Legendary'])
ax5.axis("equal")
plt.ylabel("True")
plt.xlabel("Predicted")
st.pyplot(fig5)
