from flask import Flask, render_template
import mysqldb as db

app = Flask(__name__)


@app.route('/')
def index():
    connection = db.connect('localhost', 'root', '', 'movie_rec')
    cursor = connection.cursor()
    query = "SELECT * from genres"

    cursor.execute(query)
    result = cursor.fetchall()
    for items in result:
        print(items)

@app.route('/home/<name>')
def home(name):
    return render_template("home.html", name=name)

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
