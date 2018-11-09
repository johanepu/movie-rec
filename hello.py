from flask import Flask, flash, abort, render_template, request, g, session, redirect, url_for
import MySQLdb as db
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import itertools
import math
import os
import nltk
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine
import time
from hashlib import md5
import json
from imdb import IMDb

app = Flask(__name__)
app.secret_key = 'bismillah'
app.config['DEBUG'] = True

class ServerError(Exception):pass

#preprocess
movie_data = pd.read_csv('dataset/movies.csv')
rating_info = pd.read_csv('dataset/ratings.csv')
movie_info = pd.merge(movie_data, rating_info, left_on = 'movieId', right_on = 'movieId')

def fav_movies(current_user, N):
    fav_movies = pd.DataFrame.sort_values(movie_info[movie_info.userId == current_user], ['rating'], ascending = [0]) [:N]
    return fav_movies

# Now, we will find the similarity between 2 users by using correlation
# Let's create a matrix that has the user ids on one axis and the movie title on another axis. # Let's
# Each cell will then consist of the rating the user gave to that movie.

# Lets build recommendation engine now

# We will use a neighbour based collaborative filtering model.
# The idea is to use k-nearest neighbour algorithm to find neighbours of a user
# We will use their ratings to predict ratings of a movie not already rated by a current user.
# We will represent movies watched by a user in a vector - the vector will have values for all the movies in our dataset. If a user hasn't rated a movie, it would be represented as NaN.

user_movie_rating_matrix  = pd.pivot_table(movie_info, values = 'rating', index=['userId'], columns=['movieId'])
user_movie_rating_matrix.head()

def similarity(user1, user2):
    # normalizing user1 rating i.e mean rating of user1 for any movie
    # nanmean will return mean of an array after ignore NaN values
    user1 = np.array(user1) - np.nanmean(user1)
    user2 = np.array(user2) - np.nanmean(user2)

    # finding the similarity between 2 users
    # finding subset of movies rated by both the users
    common_movie_ids = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0]
    if(len(common_movie_ids) == 0):
        return 0
    else:
        user1 = np.array([user1[i] for i in common_movie_ids])
        user2 = np.array([user2[i] for i in common_movie_ids])
        return cosine(user1, user2)

# We will now use the similarity function to find the nearest neighbour of a current user
# nearest_neighbour_ratings function will find the k nearest neighbours of the current user and
# then use their ratings to predict the current users ratings for other unrated movies
def nearest_neighbour_ratings(current_user, K):
     # Creating an empty matrix whose row index is userId and the value
    # will be the similarity of that user to the current user
    similarity_matrix = pd.DataFrame(index = user_movie_rating_matrix.index,
                                    columns = ['similarity'])
    for i in user_movie_rating_matrix.index:
        # finding the similarity between user i and the current user and add it to the similarity matrix
        similarity_matrix.loc[i] = similarity(user_movie_rating_matrix.loc[current_user],
                                             user_movie_rating_matrix.loc[i])
        # Sorting the similarity matrix in descending order
    similarity_matrix = pd.DataFrame.sort_values(similarity_matrix,
                                                ['similarity'], ascending= [0])
    # now we will pick the top k nearest neighbour
    # neighbour_movie_ratings : ratings of movies of neighbors
    # user_movie_rating_matrix : ratings of each user for every movie
    # predicted_rating : Averge where rating is NaN
    nearest_neighbours = similarity_matrix[:K]
    neighbour_movie_ratings = user_movie_rating_matrix.loc[nearest_neighbours.index]
     # This is empty dataframe placeholder for predicting the rating of current user using neighbour movie ratings
    predicted_movie_rating = pd.DataFrame(index = user_movie_rating_matrix.columns, columns = ['rating'])
    # Iterating all movies for a current user
    for i in user_movie_rating_matrix.columns:
        # by default, make predicted rating as the average rating of the current user
        predicted_rating = np.nanmean(user_movie_rating_matrix.loc[current_user])
          # j is user , i is movie
        for j in neighbour_movie_ratings.index:
            # if user j has rated the ith movie
            if(user_movie_rating_matrix.loc[j,i] > 0):# If there is some rating  # nearest_neighbours.loc[j, 'similarity']) / nearest_neighbours['similarity'].sum(): Finding Similarity score
                predicted_rating += ((user_movie_rating_matrix.loc[j,i] - np.nanmean(user_movie_rating_matrix.loc[j])) *
                                                    nearest_neighbours.loc[j, 'similarity']) / nearest_neighbours['similarity'].sum()

        predicted_movie_rating.loc[i, 'rating'] = predicted_rating

    return predicted_movie_rating

# Predicting top N recommendations for a current user
def top_n_recommendations(current_user, N):
    predicted_movie_rating = nearest_neighbour_ratings(current_user, 4)
    movies_already_watched = list(user_movie_rating_matrix.loc[current_user]
                                  .loc[user_movie_rating_matrix.loc[current_user] > 0].index)

    predicted_movie_rating = predicted_movie_rating.drop(movies_already_watched)

    top_n_recommendations = pd.DataFrame.sort_values(predicted_movie_rating, ['rating'], ascending=[0])[:N]

    top_n_recommendation_titles = movie_data.loc[movie_data.movieId.isin(top_n_recommendations.index)]

    return top_n_recommendation_titles

#checkpoint timer
@app.before_request
def before_request():
    g.request_start_time = time.time()
    g.request_time = lambda: "%.5fs" % (time.time() - g.request_start_time)

#pake SQL
@app.route('/')
def index():
    connection = db.connect('localhost', 'root', '', 'movie_rec')
    cursor = connection.cursor()
    query = "SELECT * from users"

    cursor.execute(query)
    result = cursor.fetchall()
    return render_template('db.html', data = result)

@app.route('/login', methods=['GET','POST'])
def do_admin_login():
    # if request.form['password'] == 'password' and request.form['email'] == 'admin@gmail.com':
    #     session['logged_in'] = True
    #     session['name'] = request.form['email']
    # else:
    #     flash('wrong password!')
    # return home()
    connection = db.connect('localhost', 'root', '', 'movie_rec')
    cur = connection.cursor()
    if 'logged_in' in session:
        return redirect(url_for('home'))
    error = None
    try:
        if request.method == 'POST':
            email_form  = request.form['email']
            cur.execute("SELECT COUNT(1) FROM users_detail WHERE email = %s;", [email_form]
                        )

            if not cur.fetchone()[0]:
                raise ServerError('Invalid email')

            password_form  = request.form['password']
            cur.execute("SELECT password,name,id FROM users_detail WHERE email = %s;", [email_form]
                        )

            for row in cur.fetchall():
                if password_form == row[0]:
                    session['name'] = row[1]
                    session['id'] = row[2]
                    session['logged_in'] = True
                    return redirect(url_for('home'))

            raise ServerError('Invalid password')
    except ServerError as e:
        error = str(e)

    return render_template('signin.html', error=error)

@app.route("/logout")
def logout():
    session.clear()
    return home()

#pake CSV reader
@app.route('/movie-csv')
def coba():
    movie = []
    with open('dataset/movies.csv', newline='', encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            movie.append(row)
        return render_template('db.html', data = movie)

#pake Panda dataframe
@app.route('/movie-dataframe')
def panda():
    t = request.values.get('t', 0)
    data_movies = pd.read_csv('dataset/movies.csv')
    data_movies.set_index(['movieId'], inplace=True)
    data_movies.index.name=None


    return render_template('dataframe.html',tables=[data_movies.to_html(classes='ui definition table', table_id='movie_list')],
    titles = ['na', 'Movie List'])
    time.sleep(float(t)) #just to show it works...

#pake Panda dataframe
@app.route('/rating-dataframe')
def panda_rating():
    data_ratings = pd.read_csv('dataset/ratings.csv')
    sorted_ratings = data_ratings.sort_values('userId').reset_index(drop = True)
    sorted_ratings.set_index(['userId'], inplace=True)
    sorted_ratings.index.name=None
    return render_template('dataframe.html',tables=[sorted_ratings.to_html(classes='ratings')],
    titles = ['na', 'Rating List'])

@app.route('/home/')
def home():
    t = request.values.get('t', 0)
    if not session.get('logged_in'):
        return render_template('signin.html')
    else:
        name = session.get('name')
        userId = session.get('id')
        data_movies = pd.read_csv('dataset/movies.csv')
        data_ratings = pd.read_csv('dataset/ratings.csv')
        data_links = pd.read_csv('dataset/links.csv', usecols = [0,1], dtype={'imdbId':str} )


        combine_movie_rating = pd.merge(data_ratings, data_movies, on='movieId')
        combine_movie_rating = pd.merge(combine_movie_rating, data_links, on='movieId')
        combine_movie_rating = combine_movie_rating.dropna(axis = 0, subset = ['title'])

        movie_ratingCount = (combine_movie_rating.
             groupby(by = ['title'])['rating'].
             count().
             reset_index().
             rename(columns = {'rating': 'totalRatingCount'})
             [['title', 'totalRatingCount']]
            )
        rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
        popularity_threshold = 50
        rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
        good_popular = (((rating_popular_movie.sort_values(by = 'movieId')).groupby('title')))['movieId', 'title', 'rating','totalRatingCount', 'imdbId']
        movie_list = good_popular.mean()
        movie_list['title'] = movie_list.index
        movie_list = movie_list.merge(data_links, left_on = 'movieId', right_on = 'movieId', how = 'left')
        movie_list = movie_list.as_matrix()

        movieLens = pd.merge(data_ratings, data_movies, left_on = 'movieId', right_on = 'movieId')

        def fav_movies(current_user):
            fav_movies = pd.DataFrame.sort_values(movieLens[movieLens.userId == current_user], ['rating'], ascending = [0])
            fav_movies = fav_movies[['title','genres','rating']]
            return fav_movies

        fav_movies = fav_movies(userId)
        if len(fav_movies) == 0:
            fav_movies=[{}]
        else:
            fav_movies = json.loads(fav_movies.to_json(orient='records'))

        top_rating = pd.DataFrame(movie_list, columns = ['movieId', 'avgRating','totalRatingCount','title', 'imdbId' ]).sort_values(['avgRating', 'totalRatingCount'], ascending=[False, False]).reset_index(drop = True).head(10)
        top_popular = pd.DataFrame(movie_list, columns = ['movieId', 'avgRating','totalRatingCount','title', 'imdbId' ]).sort_values(['totalRatingCount'], ascending=[False]).reset_index(drop = True).head(10)


        top_rating_json = json.loads(top_rating.to_json(orient='records'))
        top_popular_json = json.loads(top_popular.to_json(orient='records'))


        time.sleep(float(t)) #just to show it works...
        return render_template("home.html", name=name, userId=userId, fav_movies=fav_movies, top_rating=top_rating_json, top_popular=top_popular_json)
        # top_popular = movieLens.title.value_counts().head(10)

@app.route('/signin', methods=['GET', 'POST'])
def signin():

    return render_template("signin.html")
    # if request.method == 'POST':
    # 	userDetails = request.form
    #     name = userDetails['name_reg']
    #     email = userDetails['email_reg'] password = userDetails['password_reg']
    # # 	cur = mysql.connection.cursor()
	# # 	cur.execute("INSERT INTO users_detail(name, email, password) VALUES(%s, %s, %s)", (name, email, password))
	# # 	mysql.connection.commit()
	# # 	cur.close()
	# # 	return 'success'

@app.route('/register', methods=['POST'])
def register():
    # user_rating = pd.read_csv('dataset/ratings.csv')
    # id = max(user_rating.userId)
    connection = db.connect('localhost', 'root', '', 'movie_rec')
    cur = connection.cursor()
    cur.execute("SELECT MAX(id) FROM users_detail")
    max_id = cur.fetchone()
    max_id = max_id[0]
    # max_id = cur.execute("SELECT id FROM users order by id desc limit 1")
    userDetails = request.form
    name = userDetails['name_reg']
    email = userDetails['email_reg']
    password = userDetails['password_reg']
    cur.execute("INSERT INTO users_detail(id, name, email, password) VALUES(%s, %s, %s, %s)", (max_id+1, name, email, password))
    connection.commit()
    cur.close()
    return render_template('signin.html')

@app.route('/get_recommendation')
def get_recommendation():

    #preprocess
    movies = pd.read_csv('dataset/movies.csv')
    ratings = pd.read_csv('dataset/ratings.csv')
    def genre_array(str):
        return str.split('|')

    movies['genre'] = movies['genres'].apply(genre_array)
    del movies['genres']

    movie_col = list(movies.columns)
    movie_tags = movies['genre']
    tag_table = [ [token, idx] for idx, token in enumerate(set(itertools.chain.from_iterable(movie_tags)))]
    tag_table = pd.DataFrame(tag_table)
    tag_table.columns = ['Tag', 'Index']

    tag_dummy = np.zeros([len(movies), len(tag_table)])

    for i in range(len(movies)):
        for j in range(len(tag_table)):
            if tag_table['Tag'][j] in list(movie_tags[i]):
                tag_dummy[i, j] = 1

    movies = pd.concat([movies, pd.DataFrame(tag_dummy)], 1)
    movie_col.extend([string for string in tag_table['Tag']])
    movies.columns = movie_col
    del movies['genre']

    #merge dataset genre
    movieLens = pd.merge(ratings, movies, on = 'movieId')
    data = movieLens[['userId', 'movieId', 'title', 'rating']]
    movie_list = (((data.sort_values(by = 'movieId')).groupby('title')))['movieId', 'title', 'rating']
    movie_list = movie_list.mean()
    movie_list['title'] = movie_list.index
    movie_list = movie_list.as_matrix()
    movie_list = pd.DataFrame(movie_list, columns = ['movieId', 'avgRating', 'title']).sort_values('movieId').reset_index(drop = True)
    print (movie_list.shape)
    movie_list.head().sort_values(['avgRating'])

    #training
    # dividing the original data into 80:20 ratio
    utrain = (ratings[:10004])
    utest = (ratings[10004:])

    # limiting the size of the data as the longer time on larger dataset# limit
    data = ratings[:27678]
    # dividing the data into 80:20 ratio
    train_start = 1
    train_end = 160
    test_start = 161
    test_end = 200
    utrain = data[:22896]    # these are 160 users
    utest = data[22896:]    # these are 40 users

    utrain = utrain.as_matrix(columns = ['userId', 'movieId', 'rating'])
    utest = utest.as_matrix(columns = ['userId', 'movieId', 'rating'])

    train_list = []
    for i in range(train_start, train_end + 1):
        temp_list = []
        for j in range(0, len(utrain)):
            if utrain[j][0] == i:
                temp_list.append(utrain[j])
            else:
                break
        utrain = utrain[j:]
        train_list.append(temp_list)

    test_list = []
    for i in range(test_start, test_end + 1):
        temp_list = []
        for j in range(0, len(utest)):
            if utest[j][0] == i:
                temp_list.append(utest[j])
            else:
                break
        utest = utest[j:]
        test_list.append(temp_list)

    def EucledianScore(train_user, test_user):
        sum = 0
        for i in test_user:
            score = 0
            for j in train_user:
                if(int(i[1]) == int(j[1])):
                    score += ((float(i[2])-float(j[2]))*(float(i[2])-float(j[2])))
            sum = sum + score
        if sum == 0:
            sum = 10000
        return (math.sqrt(sum))

    score_list = []
    for i in range(0, test_end - train_end):
        temp_list = []
        for j in range(0, train_end):
            temp_list.append([j+1, EucledianScore(train_list[j], test_list[i])])
        score_list.append(temp_list)

    score_list = np.array(score_list)

    #get nearest
    nearest = 7
    temp_score_list = pd.DataFrame(score_list[0], columns = ['user id','Eucledian Score'])
    temp_score_list['user id'] = temp_score_list.astype('int')
    temp_score_list = temp_score_list.sort_values(by = 'Eucledian Score')
    temp_score_list = temp_score_list[:nearest][['user id', 'Eucledian Score']]

    # Taking 7 close users
    final_recommendations = []
    final_common = []
    for i in range(0, test_end - train_end):
        temp_score_list = pd.DataFrame(score_list[i], columns = ['user id','Eucledian Score'])
        temp_score_list['user id'] = temp_score_list.astype('int')
        temp_score_list = temp_score_list.sort_values(by = 'Eucledian Score')
        temp_score_list = temp_score_list[:nearest]['user id'].as_matrix()

        full_list = data[data.userId == temp_score_list[0]].movieId.as_matrix()
        for i in range(1, nearest):
            full_list = list(set().union(full_list, data[data.userId == temp_score_list[i]].movieId.as_matrix()))
        full_list = np.array(full_list)

        common_list = np.intersect1d(full_list, data[data.userId == i+test_start].movieId.as_matrix())
        common = []
        for i in common_list:
            if(movie_list[movie_list.movieId == i].as_matrix().size != 0):
                common.append(list(movie_list[movie_list.movieId == i].as_matrix()[0]))
        common = (pd.DataFrame(common,columns = ['movieId','mean rating' ,'title'])).sort_values(by = 'mean rating', ascending = False)
        final_common.append(common)

        recommendation = np.setdiff1d(full_list, common_list)
        recommendation_list = []
        for i in recommendation:
            if(movie_list[movie_list.movieId == i].as_matrix().size != 0):
                recommendation_list.append(list(movie_list[movie_list.movieId == i].as_matrix()[0]))
        recommendation = (pd.DataFrame(recommendation_list,columns = ['movieId','mean rating' ,'title'])).sort_values(by = 'mean rating', ascending = False)
        final_recommendations.append(recommendation)

    result = final_recommendations[20][:5][['mean rating' ,'title']]
    return render_template('dataframe.html',tables=[result.to_html(classes='movies')],
    titles = ['na', 'Movie List'])
        # for i in range(0, test_end - train_end):
        #     return ("\n Common for user: ", i+test_start, " -> ", final_common[i].shape[0], "\n", final_common[i][:5][['mean rating' ,'title']])

@app.route('/get_recommendation2')
def get_recommendation2():
    t = request.values.get('t', 0)
    # if request.method == 'POST':
    #     current_user = request.form['userId']


    current_user = session.get('id')
    favourite_movies = fav_movies(current_user, 5)
    recommendations = top_n_recommendations(current_user, 5)

    time.sleep(float(t)) #just to show it works...

    return render_template('result.html', fav_movies=[favourite_movies.to_html(classes='ui definition table')],
    recommendations=[recommendations.to_html(classes='ui definition table')],
    titles = ['na', 'Movie List'])

@app.route('/ratings')
def ratings():
    t = request.values.get('t', 0)
    # if request.method == 'POST':
    #     current_user = request.form['userId']

    #preprocess
    movie_data = pd.read_csv('dataset/movies.csv')
    rating_info = pd.read_csv('dataset/ratings.csv')
    movie_info = pd.merge(movie_data, rating_info, left_on = 'movieId', right_on = 'movieId')

    def fav_movies(current_user):
        fav_movies = pd.DataFrame.sort_values(movie_info[movie_info.userId == current_user], ['rating'], ascending = [0])
        fav_movies = fav_movies[['title','genres','rating']]
        return fav_movies

    def source():
        source = pd.DataFrame.sort_values(movie_data, ['movieId'], ascending = [1])
        source = source[['movieId','title']]
        return source

    current_user = session.get('id')
    fav_movies = fav_movies(current_user)
    if len(fav_movies) < 1:
        fav_movies=0
    else:
        fav_movies = json.loads(fav_movies.to_json(orient='records'))
    source = source()
    source = json.loads(source.to_json(orient='records'))

    time.sleep(float(t)) #just to show it works...

    return render_template('ratings.html', fav_movies=fav_movies, userId=current_user, source=source,
    titles = ['na', 'Movie List'])

@app.route('/insert_rating', methods=['GET','POST'])
def insert_rating():
    if request.method == 'POST':
        userId = session.get('id')
        id_input  = request.form.getlist('id[]')
        rating_input  = request.form.getlist('rating[]')

        # idarray = id_input.split(",")
        # ratingarray = rating_input.split(",")

        # import csv
        # fields=['first','second','third']
        # with open(r'name', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(fields)
        # with open('dataset/ratings.csv', 'a') as csvfile:
            # fieldnames = ['userId', 'movieId', 'rating', 'timestamp']
            # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #
            # writer.writeheader()
            # i = 0
            # while i < len(idarray):
            #     writer.writerow({'userId': userId, 'movieId': idarray[i], 'rating': ratingarray[i], 'timestamp': '123445667'})
            #     i+1

        row = [userId, id_input[0], rating_input[0], '12345567']

        with open('dataset/ratings.csv', 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            i=0
            while i < len(id_input):
                writer.writerow([userId, id_input[i], rating_input[i], '12345567'])
                i += 1
        csvFile.close()

    # return render_template('test.html',rating=id_input[1],id=rating_input[1])
    return ratings()
    # return 'succes'

if __name__ == "__main__":
    app.secret_key = 'bismillah'
    sess.init_app(app)
    app.run(debug=True)
