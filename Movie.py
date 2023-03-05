


class Movie:
    """ This class provides a way to store movie related information """
    
    def __init__(self, movieId, title, genres=[]):
        self.ID = movieId
        self.title = title
        self.genres = genres
    
    def __repr__(self):
        return f"Movie: {self.title} ({self.ID})"
    
    def __hash__(self):
        return hash(self.ID)
    
