# game_state.py
class GameState:
    def __init__(self):
        self.party = {"fighter": {"HP": 20}, "wizard": {"HP": 12}}
        self.location = "Dark Cave"
        self.history = []
    def update_from(self, narrative: str):
        # parse key events (e.g. HP changes) via simple regex or JSON tag  
        # (advanced: use embeddings & similarity search to recall past events)
        self.history.append(narrative)
