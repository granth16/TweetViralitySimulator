"""A small, hand-labeled *face-validity* benchmark.

These are **priors, not ground truth.** Each tweet is tagged with an expected
spread tier (0 = dead, 3 = viral-shaped) based on well-known engagement
patterns. A good simulator should rank them roughly in tier order and respect
the directional invariants below.

This benchmark only proves the model isn't *nonsense*. Real accuracy requires
real outcome data — loaded by the private backend through the ``storage`` seam
and scored with this same harness. Swap ``LABELED`` for a real dataset and the
metrics become meaningful predictions instead of sanity checks.
"""

from __future__ import annotations

from typing import List, Tuple

# (tweet, expected_tier) — tier 0 dead ... 3 viral-shaped.
LABELED: List[Tuple[str, int]] = [
    # --- tier 3: strong hook, contrarian/emotional, concise, no link ---
    ("Unpopular opinion: most productivity advice is just procrastination with extra steps.", 3),
    ("Nobody tells you this, but most of your problems are just decisions you keep avoiding.", 3),
    ("Stop scrolling. The best career advice I ever got was 4 words: do the scary thing.", 3),
    ("Hot take: 90% of meetings are just expensive group therapy for managers.", 3),
    ("The most underrated skill in your 20s? Being bored without reaching for your phone.", 3),
    ("I quit my $200k job to do this. Best decision I ever made. Here's why:", 3),
    # --- tier 2: solid point, some hook, shareable ---
    ("Remote work made a lot of people worse at their jobs, not better.", 2),
    ("Reading 10 pages a day adds up to about 12 books a year. Small habits win.", 2),
    ("AI won't take your job. Someone using AI will.", 2),
    ("Most people don't need more motivation. They need fewer distractions.", 2),
    ("Your network is your net worth, but only if you actually help people first.", 2),
    ("Discipline is just remembering what you actually want.", 2),
    # --- tier 1: bland, low-energy, little reason to share ---
    ("Just finished a great book about productivity. Highly recommend it.", 1),
    ("Working on some new projects this week, excited to share soon.", 1),
    ("Coffee really does make mornings better, doesn't it?", 1),
    ("Had a pretty productive day today, got a lot done.", 1),
    ("Thinking about the future of technology lately.", 1),
    ("Grateful for my team and everything we've built together.", 1),
    # --- tier 0: dead on arrival (buried links, hashtag stuffing, filler) ---
    ("Read my new blog post about productivity here: https://example.com #productivity #tips #blog", 0),
    ("Check out our latest webinar, registration link below https://example.com", 0),
    ("Good morning everyone, have a nice day!", 0),
    ("ICYMI: our Q3 newsletter is now available. Link in bio.", 0),
    ("Testing testing 123.", 0),
    ("lunch was ok i guess", 0),
]

# Directional invariants: (worse, better) — the simulator must score the second
# strictly higher. These encode causal effects, not absolute levels.
INVARIANTS: List[Tuple[str, str]] = [
    # External link suppresses reach.
    ("My favorite essay on focus is here https://example.com",
     "My favorite essay on focus changed how I work. The one idea: protect your mornings."),
    # A real hook beats a flat statement of the same idea.
    ("Sleep is important for productivity.",
     "Unpopular opinion: your productivity problem is actually a sleep problem."),
    # Concise beats rambling.
    ("I have been thinking a lot lately about how maybe people should probably try to "
     "focus more on doing fewer things and that might possibly help them.",
     "Do fewer things. Finish them. That's the whole productivity system."),
    # Hashtag stuffing hurts (stuffed version is the worse one).
    ("The best way to learn is to teach. #learning #education #growth #mindset #tips",
     "The best way to learn is to teach."),
]
