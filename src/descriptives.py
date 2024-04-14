import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from wordcloud import WordCloud


def plot_missings(missings, rotate_xticks=45, directory: str = "assets/plots/descriptives"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=missings.values, y=missings.index,
        palette='viridis', orient='h', hue=missings.index, legend=False
        )
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Columns')
    plt.savefig(os.path.join(directory, "missings.png"))
    plt.show()


def plot_ages(data, directory: str = "assets/plots/descriptives"):
    data["age_cleaned"] = pd.to_numeric(data['age'].str[:2], errors='coerce')

    overall_mean_age = round(data['age_cleaned'].median(), 2)
    chef_mean_age = round(data[data['surname_household'].notna()]['age_cleaned'].median(), 2)

    plt.figure(figsize=(10, 6))
    sns.histplot(data['age_cleaned'], color="blue", kde=True, edgecolor='black')

    plt.axvline(
        overall_mean_age, color='green', linestyle='--', linewidth=2,
        label=f'Age median global : {overall_mean_age}')
    plt.axvline(
        chef_mean_age, color='red', linestyle='--', linewidth=2,
        label=f'Age median des chefs de famille : {chef_mean_age}')

    plt.xlabel('Age')
    plt.ylabel('Effectif')
    plt.title('')
    plt.legend()
    plt.savefig(os.path.join(directory, "ages.png"))
    plt.show()


def plot_wordcloud(df, string_column: str, directory: str = "assets/plots/descriptives"):
    col_counts = df[string_column].str.lower().value_counts().to_dict()
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', mode="RGBA"
        ).generate_from_frequencies(col_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wordcloud.to_file(os.path.join(directory, "wordcloud.png"))
