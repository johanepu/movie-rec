from flask import Flask, render_template
import MySQLdb as db
import csv
import pandas as pd

app = Flask(__name__)

#pake SQL
@app.route('/')
def index():
    connection = db.connect('localhost', 'root', '', 'movie_rec')
    cursor = connection.cursor()
    query = "SELECT * from users"

    cursor.execute(query)
    result = cursor.fetchall()
    return render_template('db.html', data = result)

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
    data_movies = pd.read_csv('dataset/movies.csv')
    data_movies.set_index(['movieId'], inplace=True)
    data_movies.index.name=None
    return render_template('dataframe.html',tables=[data_movies.to_html(classes='movies')],
    titles = ['na', 'Movie List'])

#pake Panda dataframe
@app.route('/rating-dataframe')
def panda_rating():
    data_ratings = pd.read_csv('dataset/ratings.csv')
    sorted_ratings = data_ratings.sort_values('userId').reset_index(drop = True)
    sorted_ratings.set_index(['userId'], inplace=True)
    sorted_ratings.index.name=None
    return render_template('dataframe.html',tables=[sorted_ratings.to_html(classes='ratings')],
    titles = ['na', 'Rating List'])

@app.route('/home/<name>')
def home(name):
    data_movies = pd.read_csv('dataset/movies.csv')
    data_ratings = pd.read_csv('dataset/ratings.csv')
    movieLens = pd.merge(data_ratings, data_movies, left_on = 'movieId', right_on = 'movieId')
    data_merged = movieLens[['userId', 'movieId', 'title', 'rating']]
    movie_list = (((data_merged.sort_values(by = 'movieId')).groupby('title')))['movieId', 'title', 'rating']
    movie_list = movie_list.mean()
    movie_list['title'] = movie_list.index
    movie_list = movie_list.as_matrix()
    movie_list = pd.DataFrame(movie_list, columns = ['movieId', 'avgRating', 'title']).sort_values('movieId').reset_index(drop = True)
    top_rating = movie_list.sort_values(['avgRating'],ascending=False).head(10)
    top_popular = movie_list.sort_values(['avgRating'],ascending=False).head(10)
    return render_template("home.html", name=name, top_movie_tables=[top_rating.to_html(classes='top_rating')],
    pop_movie_tables=[top_popular.to_html(classes='top_popular')], titles = ['na', 'Top Rated'])
    # top_popular = movieLens.title.value_counts().head(10)
@app.route('/signin', methods=['GET', 'POST'])
def signin():
	# if request.method == 'POST':
	# 	#fetch data
	# 	userDetails  = request.form
	# 	name = userDetails['name_reg']
	# 	email = userDetails['email_reg']
	# 	password = userDetails['password_reg']
	# 	cur = mysql.connection.cursor()
	# 	cur.execute("INSERT INTO users_detail(name, email, password) VALUES(%s, %s, %s)", (name, email, password))
	# 	mysql.connection.commit()
	# 	cur.close()
	# 	return 'success'
    return render_template("signin.html")

if __name__ == "__main__":
	app.run(debug=True)
