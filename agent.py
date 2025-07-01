# agent.py
import os, json
from openai import OpenAI

OFFTOPIC_KEYWORDS = ["kubernetes", "docker", "manifest", "openai", "api_key", "grafana"]

def is_offtopic(command: str) -> bool:
    return any(kw in command.lower() for kw in OFFTOPIC_KEYWORDS)

client = OpenAI(api_key="")
from game_state import GameState
from prompts import system_prompt, player_prompt


def call_llm(messages):
    resp = client.chat.completions.create(model="gpt-4o-mini",
    messages=messages,
    temperature=0.7,
    max_tokens=500)
    return resp.choices[0].message.content

def main():
    state = GameState()
    history = [{"role":"system","content": system_prompt()}]

    while True:
        player = input("> ")
        if player.lower() in ("quit","exit"):
            break
        history.append({"role":"user","content": player_prompt(player, state)})
        reply = call_llm(history)
        print(reply)
        state.update_from(reply)
        history.append({"role":"assistant","content": reply})

if __name__ == "__main__":
    main()
