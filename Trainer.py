import random
import time
import pandas as pd
import numpy as np
from matplotlib import rcParams
from seaborn import color_palette
from sklearn.cluster import KMeans
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Memory
import logging

# Configure logging
logger = logging.getLogger('server_logger')
logger.setLevel(logging.INFO)
# Create file handler which logs down to info messages
fh = logging.FileHandler('app.log')
fh.setLevel(logging.INFO)
# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# Add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# Disable warning that is not applicable
pd.options.mode.chained_assignment = None  # default='warn'

# Read in user rating data from ratings.csv
user_ratings = pd.read_csv("Data/ratings.csv")
print(len(user_ratings))
user_ratings['user_id'] = user_ratings['user_id'].astype(str)

# Read in book data
books = pd.read_csv("Data/books.csv")

# Dictionary to allow cross-referencing book IDs and titles
id_title = {}
for i in books.itertuples():
    id_title[i[1]] = i[11]

# Dictionary to allow cross-referencing book IDs and authors
id_author = {}
for i in books.itertuples():
    id_author[i[1]] = i[8]

# Dictionary to allow cross-referencing book IDs and book cover image URLs
id_url = {}
for i in books.itertuples():
    id_url[i[1]] = i[22]

# Plot books with most ratings and display average rating
top_10_rated_books = books.sort_values(by='average_rating', ascending=False).head(10)
x = list(top_10_rated_books['ratings_count'])
y = list(top_10_rated_books['average_rating'])
labels = list(top_10_rated_books['title'])
fig, ax = plt.subplots()
ax.set_facecolor('honeydew')
ax.scatter(x, y, c='dodgerBlue')
plt.title('Top Ten Rated Books and Number of Ratings')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
for i, txt in enumerate(labels):
    xshift = .005
    if i == 1:
        ax.annotate(txt, (x[i] + xshift, y[i]), size=6, rotation=45)
    elif i == 3:
        ax.annotate(txt, (x[i] + xshift, y[i] - .003), size=6)
    elif i == 4:
        ax.annotate(txt, (x[i] + xshift, y[i]), size=6)
    elif i == 8:
        ax.annotate(txt, (x[i] + xshift, y[i] - .003), size=6)
    elif i == 9:
        ax.annotate(txt, (x[i] + xshift, y[i] + .001), size=6)
    elif i == 7:
        ax.annotate(txt, (x[i] + xshift, y[i]), size=6, rotation=90)
    else:
        ax.annotate(txt, (x[i] + xshift, y[i]), size=6)
plt.savefig("static/images/plt2.png")
plt.close('all')
palette = color_palette(n_colors=10)

# Plot books with the highest number of ratings
most_ratings = books.sort_values(by='ratings_count', ascending=False).head(10)
rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(14, 6))
y = most_ratings['title']
width = most_ratings['ratings_count']
most_viz = plt.barh(y, width, edgecolor='silver', color=palette)
plt.title('Books With the Most Ratings')
plt.xlabel('Number of Ratings (in millions)')
plt.savefig("static/images/plt3.png")
plt.close('all')

# This reduces the number of ratings by sampling randomly from the group of ratings,
# and gives 500,000 ratings.  This is useful to reduce the training time of the SVD algorithm.
# user_ratings = user_ratings[user_ratings['user_id'].map(user_ratings['user_id'].value_counts()) > 155]
# user_ratings = user_ratings[user_ratings['user_id'].map(user_ratings['user_id'].value_counts()) < 300]
user_ratings = user_ratings.sample(n=500000)
id_ratings_count = books[["average_rating", "ratings_count"]]
id_with_clusters = books[["book_id"]]

kmeans = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(id_ratings_count)

identified_clusters = kmeans.fit_predict(id_ratings_count)
data_with_clusters = id_ratings_count.copy()
data_with_clusters['Cluster'] = identified_clusters
id_with_clusters['Cluster'] = identified_clusters

plt.scatter(data_with_clusters['ratings_count'], data_with_clusters['average_rating'], c=data_with_clusters['Cluster'],
            cmap='rainbow')
plt.title('Average Rating vs. Number of Ratings for book - K-means Clustering')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.savefig("static/images/plt1.png")
plt.close('all')

# Remove the first book cluster from user data to prevent excessively rated books from entering model training data
id_with_clusters['Cluster'] = id_with_clusters['Cluster'].astype('str')
clustered_books = id_with_clusters[~id_with_clusters.Cluster.str.contains('3')]
print("Number of ratings used for SVD fit: ", len(user_ratings))


# Function to get a new user ID
def getuserID():
    new_user_ID = max(user_ratings['user_id'].astype(int)) + 1
    return new_user_ID


# Initialize array for the user's ratings
def init_rating():
    my_rating = {'user_id': [],
                 'book_id': [],
                 'rating': []}
    return my_rating


my_rating = init_rating()


def add_rating(id, book_id, rating, dataframe=pd.DataFrame):
    temp = pd.DataFrame([[id, book_id, rating]], columns=['id', 'book_id', 'rating'])
    print(temp)
    dataframe.append(temp, ignore_index=True)
    return dataframe


def clear_rating():
    del (my_rating)


# Generate new user ID
newId = getuserID()


# Function to return random book title.
# Only books with >600000 ratings are used in order to present only relatively popular books for rating.
def get_random_title(book_id):
    print("ids to check against: ", book_id)
    newbooks = books[books['ratings_count'] > 600000]
    random_id = []
    for i in range(100):
        random_id.append(newbooks.book_id.sample().item())
    random_id = random.choice(random_id)
    if random_id in book_id:
        return get_random_title(book_id)
    else:
        random_title = id_title[random_id]
        imgurl = id_url[random_id]
        return random_title, random_id, imgurl


def get_author(id):
    author = id_author[id]
    return author


# Obtain dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_ratings[['user_id', 'book_id', 'rating']], reader)
train, test = train_test_split(data, test_size=0.25)

# Initialize Singular Value Decomposition algorithm and train on training data
algo = SVD()
print("training...")
algo.fit(train)
print("complete")
full_data_trainset = data.build_full_trainset()


# Function to return top 10 predicted books
def prediction(id, full_rating):
    predictions = []

    # This section of code manipulates the data to provide more useful ratings.  The standard deviation is calculated and
    # ratings more than 2.25 standard deviations away are discarded.  This value was determined through testing.
    rating_std = books['average_rating'].std()
    rating_mean = books['average_rating'].mean()
    rating_max = rating_mean + (2.25 * rating_std)
    rating_min = rating_mean - (2.25 * rating_std)
    shortbooks = books[books['average_rating'] > rating_min]
    shortbooks = shortbooks[shortbooks['average_rating'] < rating_max]

    # Reseed the Numpy random generator using current time in integer format.
    t = time.monotonic_ns()
    last8 = int(repr(t)[-8])
    np.random.seed(last8)

    # Obtain random permutation of books and sample 250 books for ratings.  250 was chosen because it provides excellent
    # speed for the prediction algorithm and will also provide books the user likely has not seen before.
    shortbooks = shortbooks.iloc[np.random.permutation(len(shortbooks))]
    shortbooks = shortbooks.sample(n=250)

    # Predict user rating of all books in shortbooks dataframe
    for book_id in shortbooks['book_id']:
        predictions.append(algo.predict(uid=id, iid=book_id))
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Remove books the user has already rated (it is assumed they have read these books)
    user_book_id = full_rating.book_id.unique()
    for i in user_book_id:
        for j in predictions:
            if j.iid == i:
                predictions.remove(j)

    # Ensure only one book from each author is included
    authors = [id_author[pred.iid] for pred in predictions]
    for i in authors:
        count = 0
        for j in predictions:
            if id_author[j.iid] == i:
                if count == 0:
                    count = count + 1
                    pass
                else:
                    predictions.remove(j)

    # Build useful arrays
    top10ID = [pred.iid for pred in predictions[:10]]
    top10 = [id_title[pred.iid] for pred in predictions[:10]]
    top10est = [pred.est for pred in predictions[:10]]
    top10authors = [id_author[pred.iid] for pred in predictions[:10]]
    top10urls = [id_url[pred.iid] for pred in predictions[:10]]

    # Get titles of books that user has rated
    full_rating_titles = []
    for i in user_book_id:
        temp = id_title[i]
        full_rating_titles.append(temp)

    # Log user rated books and book predictions for later analysis, if desirable
    msg2 = "User input: " + "".join(str(full_rating_titles)) + "".join(str(full_rating['book_id'].to_numpy())) \
           + "".join(str(full_rating['rating'].to_numpy()))
    msg1 = "Top 10 book predictions: " + "".join(str(top10ID)) + "".join(str(top10)) + "".join(str(top10est))
    logger.info(msg1)
    logger.info(msg2)


    # Define and return all variables for display in web page
    title1 = top10[0]
    title2 = top10[1]
    title3 = top10[2]
    title4 = top10[3]
    title5 = top10[4]
    title6 = top10[5]
    title7 = top10[6]
    title8 = top10[7]
    title9 = top10[8]
    title10 = top10[9]
    author1 = top10authors[0]
    author2 = top10authors[1]
    author3 = top10authors[2]
    author4 = top10authors[3]
    author5 = top10authors[4]
    author6 = top10authors[5]
    author7 = top10authors[6]
    author8 = top10authors[7]
    author9 = top10authors[8]
    author10 = top10authors[9]
    img1 = top10urls[0]
    img2 = top10urls[1]
    img3 = top10urls[2]
    img4 = top10urls[3]
    img5 = top10urls[4]
    img6 = top10urls[5]
    img7 = top10urls[6]
    img8 = top10urls[7]
    img9 = top10urls[8]
    img10 = top10urls[9]
    return title1, title2, title3, title4, title5, title6, title7, title8, title9, title10, author1, author2, author3, \
           author4, author5, author6, author7, author8, author9, author10, img1, img2, img3, img4, img5, img6, img7, \
           img8, img9, img10
