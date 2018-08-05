
from flask import Flask
import os


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    dataset = tablib.Dataset()
    with open(os.path.join(os.path.dirname(__file__),'movies.csv')) as f:
        dataset.csv = f.read()

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def hello():
        return render_template('cover.html')

    @app.route('/movie_list')
    def movie_list():
        data = dataset.html
        return render_template('vmovielist.html', data=data)

    return app
