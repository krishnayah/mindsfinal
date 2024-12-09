import os
import scrape
from openai import APIError, OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# progress bar for df.apply
import embeddings_manager

from tqdm import tqdm
tqdm.pandas()

api_key = "oPEN AI KEY HERE"

client = OpenAI(api_key=api_key)


def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

def get_text_embedding(text):
    embedding = client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding
    return embedding


# calculate the left wing and right wing biased of a given sentence

keywords_political = [
    "Government", "Election", "Policy", "Congress", "Parliament", "President",
    "Senate", "Governor", "Democrat", "Republican", "Liberal", "Conservative",
    "Campaign", "Legislation", "Law", "Protest", "Taxes", "Supreme Court",
    "Immigration", "Rights", "Justice", "Military", "War", "Diplomacy",
    "Treaty", "Regulation", "Sanctions", "Trade", "Leadership", "Authority"
]

keywords_non_political = [
    "Entertainment", "Celebrity", "Sports", "Recipe", "Fitness", "Travel",
    "Vacation", "Weather", "Pets", "Science", "Technology", "Music",
    "Movies", "Fashion", "Art", "Food", "Gardening", "Hobby", "DIY",
    "Festival", "Wildlife", "Astronomy", "Adventure", "Fitness",
    "Health", "Books", "Culture", "Innovation", "Trends"
]

keywords_inflammatory = [
    # Negative Inflammatory Terms
    "Outrage", "Scandal", "Exposed", "Crisis", "Corruption", "Fraud",
    "Disaster", "Lies", "Betrayal", "Hypocrisy", "Greed", "Conspiracy",
    "Attack", "War", "Chaos", "Hate", "Danger", "Threat", "Fear",
    "Extreme", "Radical", "Violation", "Treason", "Illegal", "Terror",
    "Destroy", "Shocking", "Explosive", "Rebellion", "Revolt", "Insane",
    "Nightmare", "Traitor", "Deception", "Collapse", "Unprecedented",

    # Positive Inflammatory Terms
    "Victory", "Triumph", "Heroic", "Glorious", "Unstoppable", "Legendary",
    "Revolutionary", "Game-Changer", "Incredible", "Amazing", "Inspiring",
    "Transformative", "Empowering", "Groundbreaking", "Defender", "Champion",
    "Savior", "Protector", "Visionary", "Celebrated", "Historic",
    "Unwavering", "Unbreakable", "Unifying", "Epic", "Masterful",
    "Unparalleled", "Unmatched", "Rising", "Defiant"
]

keywords_neutral = [
    "Report", "Update", "Analysis", "Study", "Data", "Statistics",
    "Summary", "Overview", "Event", "Meeting", "Announcement", "Details",
    "Information", "Schedule", "Plan", "Project", "Budget", "Proposal",
    "Document", "Review", "Team", "Organization", "System", "Process",
    "Framework", "Structure", "Guidelines", "Policy", "Procedure", "Timeline",
    "Program", "Development", "Task", "Operations", "Goal", "Objective",
    "Resources", "Agreement", "Schedule", "Coordination", "Management",
    "Implementation", "Evaluation", "Status", "Update", "Neutral",
    "Fact", "Concept", "General", "Average", "Result", "Outcome"
]

keywords_positive = [
    "Happy", "Optimistic", "Positive News"
]
keywords_negative = [
    "Sad", "Pessimistic", "Negative News"
]

keyword_bins = {
    "political": keywords_political,
    "non_political": keywords_non_political,
    "inflammatory": keywords_inflammatory,
    "neutral": keywords_neutral,
    "positive": keywords_positive,
    "negative": keywords_negative
}


def calculate_bias(text):
    text_embedding = get_text_embedding(text)
    averages = {
        "political": 0,
        "non_political": 0,
        "inflammatory": 0,
        "neutral": 0,
        "positive": 0,
        "negative": 0
    }

    for bin, keywords in keyword_bins.items():
        for keyword in keywords:
            keyword_embedding = embeddings_manager.get_text_embedding(keyword)
            similarity = calculate_cosine_similarity(text_embedding, keyword_embedding)
            if averages[bin] == 0:
                averages[bin] = similarity
            else:
                averages[bin] += similarity

    for bin in averages:
        averages[bin] /= len(keyword_bins[bin])


    print (averages)

    attributes = {
        "political": "political" if averages["political"] > averages["non_political"] else "non-political",
        "inflammatory": "inflammatory" if averages["inflammatory"] > averages["neutral"] else "neutral",
        "sentiment": "positive" if averages["positive"] > averages["negative"] else "negative"
    }

    attributes["bias"] = bias_heuristic(attributes)


    return attributes




def bias_heuristic(values):
    sum = 0
    # highest weight is given to text that matches political, inflammatory and negative
    if values["political"] == "political text":
        sum += 1
    if values["inflammatory"] == "inflammatory":
        sum += 1
    if values["sentiment"] == "negative":
        sum += 1

    return sum


print(calculate_bias("I really hate the government, they are so corrupt and greedy."))
print(calculate_bias("I love the government, they are so helpful and kind."))



scrape.scrape_fox()
# create a pandas dataframe from the text file
# text is just the headline
df = pd.read_csv("fox_headlines.txt", header=None, sep="~")
df.columns = ['headline']

df = df.sample(10)


df = pd.concat([df, df['headline'].progress_apply(calculate_bias).apply(pd.Series)], axis=1)
print(df)
df.to_csv("fox_headlines_bias.csv", index=False)

scrape.scrape_cnn()
df = pd.read_csv("cnn_headlines.txt", header=None, sep="~")
df.columns = ['headline']
df = df.sample(10)
df = pd.concat([df, df['headline'].progress_apply(calculate_bias).apply(pd.Series)], axis=1)
print(df)
df.to_csv("cnn_headlines_bias.csv", index=False)