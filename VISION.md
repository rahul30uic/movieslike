# Movieslike — the anti-scroll movie picker

## The problem
Streaming services cause choice paralysis. You scroll endlessly, and even after
picking something, you can't invest in it because "there might be something
better." The wall of options is the enemy.

## The insight
People usually can't articulate what they want to watch — but they reliably
know what they *don't* want. Preference is easier to express as reaction than
as description.

## The product
Never show a wall. Show **5 movies**.

The user reacts to them two ways:

1. **In words** — a conversation with an LLM ("like X but newer, with comedy",
   "no comedy, I want batshit-crazy characters"). The LLM refines the picks
   and, in the process, helps the user discover their own head-space.
2. **Where words fail, in vibes** — the user picks between 2–3 contrasting
   mood images (sourced from a Reddit corpus of "movies that feel like this"
   posts, embedded into a vibe-space). A few rounds of picks triangulate the
   user's head-space without them ever having to describe it.

Each round *narrows* the choice — it never reshuffles it. The session ends
with the user committing to **one movie**, confident nothing better was one
more scroll away.

## North star
Minimize time-from-opening-the-app-to-actually-watching. Not engagement,
not time-in-app. A great session is a short one that ends in a movie the
user finishes.
