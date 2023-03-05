

class User:

    def __init__(self, userId):
        self.ID = userId
        # self.movie_ratings = movie_ratings #doesn't work because it creates a reference to the same dictionary
        self.movie_ratings = {}
        self.added_movies = 0

    def add_movie_rating(self, movieId, rating):
        if not isinstance(movieId, int):
            raise TypeError("movieId must be an integer")
        if movieId in self.movie_ratings.keys():
            print(f"User {self.ID} already rated movie {movieId}")
            return
        self.added_movies += 1
        print(f"User {self.ID} rated movie {movieId} with a rating of {rating}")
        self.movie_ratings[movieId] = rating

    def __hash__(self):
        return hash(self.ID)
    
    def __repr__(self):
        return f"User {self.ID} rated {len(self.movie_ratings)} movies"

