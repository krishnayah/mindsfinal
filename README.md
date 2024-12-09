# Phase 1
1. This prototype scrapes Fox news, and adds all the values to a CSV
2. From there, I get the embedding vectors of all the headlines, and compare them against the following vectors
	1. Vector inference of "The text is inflammatory"
	2. Vector inference of "The text is inflammatory"
3. **The issue with this model is that it returns too many false positives (positive meaning inflammatory, in this scenario.)**

Take this for instance:

`"Jonathan Majors & Meagan Good Engaged (NEW: 8AM),inflammatory"`

What does 'neutral' even mean in this context? It's comparing the content of the headlines to the words 'neutral' and 'inflammatory' themselves, which doesn't mean much on its own. 

**Next proposed solution:** Instead of using the inference of 'The text is inflammatory' or 'The text is neutral,' I'll use statements like:
- "inflammatory": "inflammatory and/or biased. This text is likely to sway opinions"
- "neutral": "neutral text, doesn't contain any bias nor try to sway opinion"

*Alternate Solution: I could create entire sets of inflammatory headlines and compare them, creating a score of how similar they are to a set of inflammatory/biased headlines, and a score of how similar they are to a set of neutral, normal headlines. This score determines the final ranking of the headline*

# Phase 2

After implementing the idea above, I still encounter some issues: "Netflix 'cannot get away with' production issues for NFL games, ex-star Shawne Merriman warns",inflammatory

Although this is inflammatory news, it isn't politically charged. And this is an issue with such text embeddings. I can't really compare the description of "this is text is biased politically and inflammatory," without risking high levels of false positives or false negatives.
In this case, a new idea would be to assign multiple traits to a certain headline, eventually ranking it on a score. More of this will be described in the Phase 3 implementation.

# Phase 3

Here I decided to create a heuristic for evaluating the biases of a text.

Rather than trusting a single metric, such as "biased media," I instead created a Python dictionary featuring tags, and also what that text may describe or look like. 

Here's what that looks like

```
labels = {
		  "inflammatory": "inflammatory text, tries to sway opinions or incite
strong emotions. Contains political bias.",
 "neutral": "neutral text, doesn't contain any bias nor try to sway
opinions. Simply just facts, or uplifting news.",

 "positive": "positive text, tries to uplift the reader or make them feel
happy about good news in the world",
 "neutral_sentiment": "doesn't contain any sentiment, just a statement or
fact. ",
 "negative": "negative text, tries to make the reader feel sad or angry
about the news in the world",

 "political text": "political text, contains bias towards a political
party or ideology. refers politicians or political terminology",
 "non-political text": "non-political text, doesn't contain any political
bias or terminology. Just a statement, new fact, or uplifting news.",

}
```


For each of these labels, I assign them an embedding using the OpenAI API

```python
def get_text_embedding(text): 
	embedding = client.embeddings.create(input=text, model="text-embedding3-large").data[0].embedding 
	return embedding
```


Now, I have a set of vector embeddings, essentially points in space that represent each text. These vector embeddings key to each sentence, and encode pre-trained data about it's meaning, semantics, and potential related words/phrases. 

For each headline, I also retrieve the embedding.

```python
def calculate_bias(text): 
	text_embedding = get_text_embedding(text) 
	bias_values = {} 
	for label, label_embedding in label_embeddings.items(): 
		bias_values[label] = calculate_cosine_similarity(text_embedding, label_embedding) 
	values = {}
	values["inflammatory"] = "inflammatory" if bias_values["inflammatory"] > bias_values["neutral"] else "neutral" 
	values["political"] = "political text" if bias_values["political text"] > bias_values["non-political text"] else "non-political text" 
	max(bias_values["positive"], bias_values["neutral_sentiment"], bias_values["negative"]) 
	if max_value == bias_values["positive"]: 
		values["sentiment"] = "positive" 
	elif max_value == bias_values["neutral_sentiment"]: 
		values["sentiment"] = "neutral_sentiment" 
	else: 
		values["sentiment"] = "negative"
	return bias_heuristic(values) # which sums up a total of negative, political and inflammatory.
```


And for a bit of clarification: 
```python
from sklearn.metrics.pairwise import cosine_similarity 
def calculate_cosine_similarity(vector1, vector2): 
	return cosine_similarity([vector1], [vector2])[0][0]
```

This code is our helper function that returns the distance between the two vectors vector1 being the vector of our text, and vector2 being the vector of a label, such as 'This text is inflammatory.'

I sum up the values, and then use the 'bias_heuristic' function to calculate a score of how biased it is.

```python
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
```


Running the tests involves this follow scraper snippet: 

```python
scrape.scrape_fox() # create a pandas dataframe from the text file # text is just the headline 
df = pd.read_csv("fox_headlines.txt", header=None, sep="~") 
df.columns = ['headline'] 
df['bias'] = df['headline'].progress_apply(calculate_bias) 
# save the dataframe to a csv file df.to_csv("fox_headlines.csv", index=False)
```

## Results Breakdown:

```
"Percival Everett, Jason De León selected as winners of the 2024 National Book Awards",1 Chief opponent to Uganda's president appears in court days after going missing,2 Japan says it will watch China’s military activity after Beijing admits violating Japanese airspace,2 SEAN HANNITY: The mask is coming off the Democrats,3 JESSE WATTERS: The world is ready to move on from Biden - even Democrats,2
```

 One issue I noticed, is that the scores are rarely zero– and this can be further fixed by not just creating general blanket statements, but rather also examples of inflammatory media. My idea is that, for negative media, I could create a dictionary with sad headlines, and then compare the average similarity against all those headlines.

I began trying different phrases to pinpoint *why* my code was returning such high scores, and it turns out,  the phrase 'I love puppies, ' for some reason, returned 'political' and 'inflammatory.' 

It was clear to me that the blanket statements, such as 'this text is inflammatory,' weren't really something you could evaluate semantically. Inflammatory text can take a lot of different forms, it requires further thinking than just 'comparing' words, and because of that, using such sentence blanket statements were inefficient and yielded not great results.


# Phase 4

For the final idea, I realized something: I could create a set of keywords, such as "uplifting," "optimistic," and "hopeful." These keywords would then be evaluated compared to the existing headline, and then their average values can be taken. This average will be compared against words like "negative," and "pessimistic."


I rewrote the program to use these following sets of keywords:

```python
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
```


And from there, I created a script to gather the inferences and average them for each keyword.

```python
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
```

From there, I utilize the averages to assign each headline a set of attributes, whether it was political or not, inflammatory or neutral, etc. These are done by comparing the averages of each. A higher average similarity for the `political` keywords than the `non_political` keywords will result in an attribute of `political: "political"`
```python
attributes = {  
    "political": "political" if averages["political"] > averages["non_political"] else "non-political",  
    "inflammatory": "inflammatory" if averages["inflammatory"] > averages["neutral"] else "neutral",  
    "sentiment": "positive" if averages["positive"] > averages["negative"] else "negative"  
}
```

From there, I calculate the bias heuristic the same as previously mentioned before. Except, this time, I add the total sum *to* the attributes, meaning the final product may look something like this:
```json
attributes: {
"political": "non-political",
"inflammatory": "neutral",
"sentiment": "negative",
"bias": 1
}
```

And later on, these attributes are appended to the Pandas dataframe, meaning I can see not only just the score for each headline, but also what tags my program attributed to them.


### Cleaning up the code: Caching inferences for cost efficiency.

It was incredibly inefficient to have to query the OpenAI API for that massive list of keywords everytime I booted up the program. To solve this, I wrote a file called `embeddings_manager.py` which managed the embeddings and cached them all. 

The code for this was relatively simple, and looked a bit like this: 

```python
def get_text_embedding(text):  
    if text in embeddings_file:  
        print("found in file")  
        return embeddings_file[text]  
    else:  
        try:  
            print("not found in file, creating new")  
            embedding = client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding  
            embeddings_file[text] = embedding  
            with open(filename, "w") as f:  
                json.dump(embeddings_file, f)  
            return embedding  
        except APIError as e:  
            print(e)  
            return None
```

This code essentially ensured I stored an offline copy of my embeddings for keywords, such as 'Happy,' which I would end up using incredibly often.

I made the choice not to store the embeddings for headlines, as I wouldn't be accessing them as often and didn't want to waste file storage space for those embeddings. Obviously, on a larger scale, if this was implemented for hundreds of users, it would be viable to cache headlines too with a 2-3 day window, as users would be accessing the same sites and seeing the same headlines multiple times.


# Results & Conclusion of final prototype

I was thoroughly happy with these results. 

Text like: 
`NBC called out for 'selectively omitting key words' from Constitution in Trump interview,political,inflammatory,negative,3` tests highly positive for bias, whereas text like `8 great gifts for food and wine lovers this holiday season,non-political,neutral,positive,0` rightfully tests negative.

These two headlines were from FOX News, but it also works on CNN:
`Healthy soup recipes that are cozy and comforting,non-political,neutral,positive,0` and `A health insurance CEO is dead. Not everyone is sad,political,inflammatory,negative,2`


Media bias is something that has completely engulfed American politics & the wellbeing of citizens nation-wide. By doing more analysis and research into how bias can be measured and what can be done to mitigate bias, we can push for a better, truthful society. News agencies must be held accountable for their polarization of American politics. 

This program isn't meant to evaluate biases on a large scale. Rather, such a bias heuristic using text embeddings is a cheap and cost effective way for *anyone* to tinker around with utilizing computing to detect biases within text, and to learn more about what really makes a text biased. Programs like this can be used to generate training data for more robust, more concrete models that can be efficiently applied en masse.

The code for this is open sourced on https://github.com/krishnayah/mindsfinal
