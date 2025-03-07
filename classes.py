class Pokemon:
    def __init__(self, name):
        self.name = name
        self.moves = []

    def get_name(self):
        return self.name
    
    def add_move(self, move): 
        self.moves.append(move)
    
    def get_move_count(self):
        return len(self.moves)
        

class Team:
    def __init__(self, type):
        self.pokemon = []
        self.type = type
    
    def add_pokemon(self, pokemon):
        self.pokemon.append(pokemon)
    
    def get_pokemon_count(self):
        return len(self.pokemon)
    
    def has_pokemon(self, pokemon_name):
        return any(p.get_name() == pokemon_name for p in self.pokemon)
    
    def print_team(self):
        members = []
        for pokemon in self.pokemon:
            members.append(pokemon.get_name())
        print(members)