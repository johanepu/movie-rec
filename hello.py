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
from scipy.spatial.distance import correlation
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
data_links = pd.read_csv('dataset/links.csv', usecols = [0,1], dtype={'imdbId':str} )
movie_info = pd.merge(movie_data, rating_info, left_on = 'movieId', right_on = 'movieId')
user_movie_rating_matrix  = pd.pivot_table(movie_info, values = 'rating', index=['userId'], columns=['movieId'])

def fav_movies(current_user):
    fav_movies = pd.DataFrame.sort_values(movie_info[movie_info.userId == current_user], ['rating'], ascending = [0])
    return fav_movies

def cek_movie(user1, user2):
    # normalizing user1 rating i.e mean rating of user1 for any movie
    # nanmean will return mean of an array after ignore NaN values
    array1 = np.array(user1)
    array2 = np.array(user2)
    common_movie_ids = [i for i in range(len(user1)) if array1[i] > 0 and math.isnan(array1[i]) != True and array2[i] > 0 and math.isnan(array2[i]) != True]
    return common_movie_ids

def rating_per_user(user1):
    array_user = np.array(user1)
    ids = user_movie_rating_matrix.columns.tolist()
    rated_movies = [ids[i] for i in range(len(user1)) if array_user[i] > 0 and math.isnan(array_user[i]) != True]
    # percentage = len(rated_movies)/len(user1)
    return rated_movies

def cek_common(user1, user2):
    # normalizing user1 rating i.e mean rating of user1 for any movie
    # nanmean will return mean of an array after ignore NaN values
    ids = user_movie_rating_matrix.columns.tolist()
    array1 = np.array(user1)
    array2 = np.array(user2)
    common_movie_ids = [ids[i] for i in range(len(user1)) if array1[i] > 0 and math.isnan(array1[i]) != True and array2[i] > 0 and math.isnan(array2[i]) != True]
    return common_movie_ids

def pearson_similarity(user1, user2):
    common_movie_ids = cek_common(user1, user2)
    # finding the similarity between 2 users
    # finding subset of movies rated by both the users
    if(len(common_movie_ids) >= 1):
        vector1 = np.array([user1.loc[i] for i in common_movie_ids])
        vector2 = np.array([user2.loc[i] for i in common_movie_ids])
        return 1-correlation(vector1, vector2)
    elif(len(common_movie_ids) < 1):
        return 0

def pearson_p_similarity(user1, user2, t):
    common_movie_ids = cek_common(user1, user2)
    # normalizing user1 rating i.e mean rating of user1 for any movie
    # nanmean will return mean of an array after ignore NaN values
    # finding the similarity between 2 users
    # finding subset of movies rated by both the users
    if(len(common_movie_ids) >= t*len(rating_per_user(user1))):
        vector1 = np.array([user1.loc[i] for i in common_movie_ids])
        vector2 = np.array([user2.loc[i] for i in common_movie_ids])
        return 1-correlation(vector1, vector2)
    elif(len(common_movie_ids) < 1*len(rating_per_user(user1))):
        return 0

def cosine_similarity(user1, user2):
    common_movie_ids = cek_common(user1, user2)
    # finding the similarity between 2 users
    # finding subset of movies rated by both the users
    if(len(common_movie_ids) >= 1):
        vector1 = np.array([user1.loc[i] for i in common_movie_ids])
        vector2 = np.array([user2.loc[i] for i in common_movie_ids])
        return 1-cosine(vector1, vector2)
    elif(len(common_movie_ids) < 1):
        return 0

def cosine_p_similarity(user1, user2, t):
    common_movie_ids = cek_common(user1, user2)
    # normalizing user1 rating i.e mean rating of user1 for any movie
    # nanmean will return mean of an array after ignore NaN values
    # finding the similarity between 2 users
    # finding subset of movies rated by both the users
    if(len(common_movie_ids) >= t*len(rating_per_user(user1))):
        vector1 = np.array([user1.loc[i] for i in common_movie_ids])
        vector2 = np.array([user2.loc[i] for i in common_movie_ids])
        return 1-cosine(vector1, vector2)
    elif(len(common_movie_ids) < 1*len(rating_per_user(user1))):
        return 0

similarity_matrix = pd.DataFrame(index = user_movie_rating_matrix.index,
                                columns = ['similarity','common_movies'])

movies_already_watched = user_movie_rating_matrix
global nearest_neighbours
def pearson_p_prediction(current_user, K):
    global similarity_matrix
    nearest_neighbours = similarity_matrix[:K]
    if(nearest_neighbours['similarity'].sum()==0):
        return None
    pearson_p_prediction.table_neighbours = nearest_neighbours
    neighbour_movie_ratings = user_movie_rating_matrix.loc[nearest_neighbours.index]
     # This is empty dataframe placeholder for predicting the rating of current user using neighbour movie ratings
    predicted_movie_rating = pd.DataFrame(index = user_movie_rating_matrix.columns, columns = ['p_rating'])
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

        predicted_movie_rating.loc[i, 'p_rating'] = predicted_rating

    global movies_already_watched
    predicted_movie_rating = predicted_movie_rating.drop(movies_already_watched)
    movie_ratingCount = (movie_info.
     groupby(by = ['movieId'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['movieId', 'totalRatingCount']]
    )
    rating_with_totalRatingCount = predicted_movie_rating.merge(movie_ratingCount, left_on = 'movieId', right_on = 'movieId', how = 'left')
    # rating_w_total = predicted_movie_rating.merge(movie_list, left_on = 'movieId', right_on = 'movieId', how = 'left')
    return rating_with_totalRatingCount

# Predicting top N recommendations for a current user
def top_n_recommendations(current_user, N):
    predicted_movie_rating = pearson_p_prediction(current_user, N)
    predicted_movie_rating = pd.DataFrame.sort_values(predicted_movie_rating,
                                                ['p_rating','totalRatingCount'], ascending= [0,0])[:10]

    # top_n_recommendations = pd.DataFrame.sort_values(predicted_movie_rating, ['p_rating'], ascending=[0])[:N]

    top_n_recommendation_titles = movie_data.loc[movie_data.movieId.isin(predicted_movie_rating.movieId)]
    top_n_recommendation_titles = top_n_recommendation_titles.merge(predicted_movie_rating, left_on = 'movieId', right_on = 'movieId', how = 'left')
    top_n_recommendation_titles = pd.DataFrame.sort_values(top_n_recommendation_titles, ['p_rating','totalRatingCount'], ascending=[0,0])
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
    movie_infos.set_index(['movieId'], inplace=True)
    movie_info.index.name=None


    return render_template('dataframe.html',tables=[movie_info.to_html(classes='ui definition table', table_id='movie_list')],
    titles = ['na', 'Movie List'])
    time.sleep(float(t)) #just to show it works...

#pake Panda dataframe
@app.route('/rating-dataframe')
def panda_rating():
    sorted_ratings = rating_info.sort_values('userId').reset_index(drop = True)
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
        rating_info = pd.read_csv('dataset/ratings.csv')
        movie_info = pd.merge(movie_data, rating_info, left_on = 'movieId', right_on = 'movieId')
        combine_movie_rating = pd.merge(movie_info, data_links, on='movieId')
        combine_movie_rating = combine_movie_rating.dropna(axis = 0, subset = ['title'])

        global similarity_matrix
        similarity_matrix = similarity_matrix.drop(userId)
        for i in user_movie_rating_matrix.index:
            # finding the similarity between user i and the current user and add it to the similarity matrix
            similarity_matrix.loc[i,'similarity'] = pearson_p_similarity(user_movie_rating_matrix.loc[userId],user_movie_rating_matrix.loc[i],0.3)
            similarity_matrix.loc[i,'common_movies'] = len(cek_common(user_movie_rating_matrix.loc[userId],user_movie_rating_matrix.loc[i]))
            # Sorting the similarity matrix in descending order
        similarity_matrix = pd.DataFrame.sort_values(similarity_matrix,
                                                    ['similarity'], ascending= [0])
        global movies_already_watched
        movies_already_watched = list(user_movie_rating_matrix.loc[userId]
                                      .loc[user_movie_rating_matrix.loc[userId] > 0].index)

        movie_ratingCount = (combine_movie_rating.
             groupby(by = ['title'])['rating'].
             count().
             reset_index().
             rename(columns = {'rating': 'totalRatingCount'})
             [['title', 'totalRatingCount']]
            )
        rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
        popularity_threshold = 70
        rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
        good_popular = (((rating_popular_movie.sort_values(by = 'movieId')).groupby('title')))['movieId', 'title', 'rating','totalRatingCount', 'imdbId']
        movie_list = good_popular.mean()
        movie_list['title'] = movie_list.index
        movie_list = movie_list.merge(data_links, left_on = 'movieId', right_on = 'movieId', how = 'left')
        movie_list = movie_list.as_matrix()

        user_fav = fav_movies(userId)
        if len(user_fav) == 0:
            user_fav=[{}]
        else:
            user_fav = pd.merge(user_fav, data_links, on='movieId')
            user_fav = json.loads(user_fav.to_json(orient='records'))

        top_rating = pd.DataFrame(movie_list, columns = ['movieId', 'avgRating','totalRatingCount','title', 'imdbId' ]).sort_values(['avgRating', 'totalRatingCount'], ascending=[False, False]).reset_index(drop = True).head(10)
        top_popular = pd.DataFrame(movie_list, columns = ['movieId', 'avgRating','totalRatingCount','title', 'imdbId' ]).sort_values(['totalRatingCount'], ascending=[False]).reset_index(drop = True).head(10)


        top_rating_json = json.loads(top_rating.to_json(orient='records'))
        top_popular_json = json.loads(top_popular.to_json(orient='records'))


        time.sleep(float(t)) #just to show it works...
        return render_template("home.html", name=name, userId=userId, fav_movies=user_fav, top_rating=top_rating_json, top_popular=top_popular_json)
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


@app.route('/get_recommendation2')
def get_recommendation2():
    t = request.values.get('t', 0)
    # if request.method == 'POST':
    #     current_user = request.form['userId']


    current_user = session.get('id')
    # favourite_movies = fav_movies(current_user, 5)
    recommendations = top_n_recommendations(current_user, 5)

    recommendations = pd.merge(recommendations, data_links, on='movieId')
    recommendations = json.loads(recommendations.to_json(orient='records'))

    nearest = pearson_p_prediction.table_neighbours
    nearest = nearest.iloc[1:]
    top_id = nearest.reset_index().loc[0][0]
    top_fav = fav_movies(top_id)
    top_fav = json.loads(top_fav[:5].to_json(orient='records'))
    neighbours = json.loads(nearest.reset_index().to_json(orient='records'))
    time.sleep(float(t)) #just to show it works...

    return render_template('result.html',
    recommendations=recommendations, neighbours=neighbours, top_fav=top_fav,
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
