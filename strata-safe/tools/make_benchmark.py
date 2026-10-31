#!/usr/bin/env python3
"""Generate benchmark/loop-eval-v1.jsonl — a fixed, authored 200-item multiple-choice set
for the 3090 looped-LM evaluation (gpu_jobs/looped-eval.job.yaml).

Why an authored benchmark instead of GSM8K/MMLU:
  * no network and no committed third-party raw data in the GPU lane;
  * the paper's own base-model tables (Tables 10-11: MMLU/ARC-C/Winogrande) are multiple-choice
    scored by likelihood, so likelihood-scored MCQ is a faithful, decoding-free protocol;
  * we can control the reasoning/storage split: 100 multi-step arithmetic items (knowledge
    manipulation) + 100 single-hop factual items (knowledge storage).

Deterministic: same items for the same seed, rebuilt on demand. The .jsonl is gitignored
(no committed raw data); tests and tools/looped_eval.py rebuild it in memory when absent.
Usage:
    python3 tools/make_benchmark.py            # writes benchmark/loop-eval-v1.jsonl
    python3 tools/make_benchmark.py --check    # determinism check; checks the file if present
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "loop-eval-v1.jsonl"
SEED = 20261002


# --------------------------------------------------------------------------------------- facts
# (question, correct, [three distractors]) — public, non-personal facts only.
FACTS = [
    ("What is the capital of France?", "Paris", ["Lyon", "Marseille", "Nice"]),
    ("What is the capital of Germany?", "Berlin", ["Munich", "Hamburg", "Cologne"]),
    ("What is the capital of Italy?", "Rome", ["Milan", "Naples", "Turin"]),
    ("What is the capital of Spain?", "Madrid", ["Barcelona", "Seville", "Valencia"]),
    ("What is the capital of Portugal?", "Lisbon", ["Porto", "Braga", "Coimbra"]),
    ("What is the capital of the Netherlands?", "Amsterdam", ["Rotterdam", "Utrecht", "The Hague"]),
    ("What is the capital of Belgium?", "Brussels", ["Antwerp", "Ghent", "Bruges"]),
    ("What is the capital of Austria?", "Vienna", ["Salzburg", "Graz", "Linz"]),
    ("What is the capital of Switzerland?", "Bern", ["Zurich", "Geneva", "Basel"]),
    ("What is the capital of Sweden?", "Stockholm", ["Gothenburg", "Malmo", "Uppsala"]),
    ("What is the capital of Norway?", "Oslo", ["Bergen", "Trondheim", "Stavanger"]),
    ("What is the capital of Denmark?", "Copenhagen", ["Aarhus", "Odense", "Aalborg"]),
    ("What is the capital of Finland?", "Helsinki", ["Espoo", "Tampere", "Turku"]),
    ("What is the capital of Poland?", "Warsaw", ["Krakow", "Gdansk", "Wroclaw"]),
    ("What is the capital of Czechia?", "Prague", ["Brno", "Ostrava", "Plzen"]),
    ("What is the capital of Hungary?", "Budapest", ["Debrecen", "Szeged", "Miskolc"]),
    ("What is the capital of Greece?", "Athens", ["Thessaloniki", "Patras", "Heraklion"]),
    ("What is the capital of Ireland?", "Dublin", ["Cork", "Galway", "Limerick"]),
    ("What is the capital of Romania?", "Bucharest", ["Cluj-Napoca", "Timisoara", "Iasi"]),
    ("What is the capital of Bulgaria?", "Sofia", ["Plovdiv", "Varna", "Burgas"]),
    ("What is the capital of Croatia?", "Zagreb", ["Split", "Rijeka", "Osijek"]),
    ("What is the capital of Serbia?", "Belgrade", ["Novi Sad", "Nis", "Kragujevac"]),
    ("What is the capital of Ukraine?", "Kyiv", ["Lviv", "Odesa", "Kharkiv"]),
    ("What is the capital of Turkey?", "Ankara", ["Istanbul", "Izmir", "Bursa"]),
    ("What is the capital of Russia?", "Moscow", ["Saint Petersburg", "Novosibirsk", "Kazan"]),
    ("What is the capital of Egypt?", "Cairo", ["Alexandria", "Giza", "Luxor"]),
    ("What is the capital of Morocco?", "Rabat", ["Casablanca", "Marrakesh", "Fes"]),
    ("What is the capital of Kenya?", "Nairobi", ["Mombasa", "Kisumu", "Nakuru"]),
    ("What is the capital of Nigeria?", "Abuja", ["Lagos", "Kano", "Ibadan"]),
    ("What is the capital of South Africa?", "Pretoria", ["Cape Town", "Durban", "Johannesburg"]),
    ("What is the capital of Ethiopia?", "Addis Ababa", ["Dire Dawa", "Mekelle", "Gondar"]),
    ("What is the capital of Canada?", "Ottawa", ["Toronto", "Montreal", "Vancouver"]),
    ("What is the capital of Mexico?", "Mexico City", ["Guadalajara", "Monterrey", "Puebla"]),
    ("What is the capital of Brazil?", "Brasilia", ["Rio de Janeiro", "Sao Paulo", "Salvador"]),
    ("What is the capital of Argentina?", "Buenos Aires", ["Cordoba", "Rosario", "Mendoza"]),
    ("What is the capital of Chile?", "Santiago", ["Valparaiso", "Concepcion", "Antofagasta"]),
    ("What is the capital of Peru?", "Lima", ["Cusco", "Arequipa", "Trujillo"]),
    ("What is the capital of Colombia?", "Bogota", ["Medellin", "Cali", "Cartagena"]),
    ("What is the capital of Japan?", "Tokyo", ["Osaka", "Kyoto", "Nagoya"]),
    ("What is the capital of South Korea?", "Seoul", ["Busan", "Incheon", "Daegu"]),
    ("What is the chemical symbol for gold?", "Au", ["Ag", "Fe", "Cu"]),
    ("What is the chemical symbol for silver?", "Ag", ["Au", "Sn", "Pb"]),
    ("What is the chemical symbol for iron?", "Fe", ["Ir", "In", "Fr"]),
    ("What is the chemical symbol for copper?", "Cu", ["Co", "Cr", "Ca"]),
    ("What is the chemical symbol for sodium?", "Na", ["So", "Sn", "Ni"]),
    ("What is the chemical symbol for potassium?", "K", ["P", "Po", "Pt"]),
    ("What is the chemical symbol for helium?", "He", ["H", "Hf", "Ho"]),
    ("What is the chemical symbol for neon?", "Ne", ["N", "Ni", "Nd"]),
    ("What is the chemical symbol for lead?", "Pb", ["Le", "Ld", "Pt"]),
    ("What is the chemical symbol for tin?", "Sn", ["Ti", "Tn", "Sb"]),
    ("What is the chemical symbol for mercury?", "Hg", ["Me", "Mc", "Mg"]),
    ("What is the chemical symbol for tungsten?", "W", ["Tg", "Tu", "Ts"]),
    ("What is the chemical symbol for uranium?", "U", ["Ur", "Un", "Y"]),
    ("What is the chemical symbol for calcium?", "Ca", ["Cl", "C", "Cd"]),
    ("What is the chemical symbol for magnesium?", "Mg", ["Mn", "Mo", "Md"]),
    ("What is the chemical symbol for zinc?", "Zn", ["Zi", "Zr", "Z"]),
    ("What is the chemical symbol for nickel?", "Ni", ["N", "Ne", "Nb"]),
    ("What is the chemical symbol for chlorine?", "Cl", ["Ch", "C", "Cr"]),
    ("What is the chemical symbol for oxygen?", "O", ["Ox", "Og", "Os"]),
    ("What is the chemical symbol for nitrogen?", "N", ["Ni", "Ng", "No"]),
    ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Earth", "Mars"]),
    ("Which planet is known as the Red Planet?", "Mars", ["Jupiter", "Venus", "Saturn"]),
    ("Which planet has the most prominent ring system?", "Saturn", ["Jupiter", "Neptune", "Uranus"]),
    ("What force keeps the planets in orbit around the Sun?", "Gravity", ["Magnetism", "Friction", "Buoyancy"]),
    ("What is the largest planet in the Solar System?", "Jupiter", ["Saturn", "Neptune", "Earth"]),
    ("What is the approximate speed of light in vacuum?", "300,000 km/s", ["300 km/s", "3,000 km/s", "30,000,000 km/s"]),
    ("Which gas do plants absorb for photosynthesis?", "Carbon dioxide", ["Oxygen", "Nitrogen", "Hydrogen"]),
    ("Which organelle is known as the powerhouse of the cell?", "Mitochondria", ["Nucleus", "Ribosome", "Golgi apparatus"]),
    ("What is the chemical formula of water?", "H2O", ["CO2", "O2", "H2O2"]),
    ("How many chromosomes do humans normally have?", "46", ["23", "44", "48"]),
    ("What is the boiling point of water at sea level in Celsius?", "100", ["0", "50", "212"]),
    ("Which blood type is known as the universal donor?", "O negative", ["AB positive", "A positive", "B negative"]),
    ("What is the largest organ of the human body?", "Skin", ["Liver", "Lungs", "Intestine"]),
    ("Which vitamin does skin produce in sunlight?", "Vitamin D", ["Vitamin A", "Vitamin C", "Vitamin B12"]),
    ("What is the smallest prime number?", "2", ["1", "3", "0"]),
    ("What is the value of pi to two decimal places?", "3.14", ["3.41", "2.72", "1.62"]),
    ("Who proposed the theory of general relativity?", "Albert Einstein", ["Isaac Newton", "Niels Bohr", "Max Planck"]),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen", ["Oxygen", "Carbon dioxide", "Argon"]),
    ("What is the hardest naturally occurring substance?", "Diamond", ["Quartz", "Granite", "Steel"]),
    ("How many bones are in the adult human body?", "206", ["186", "226", "246"]),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare", ["Charles Dickens", "Jane Austen", "Oscar Wilde"]),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["Michelangelo", "Raphael", "Vincent van Gogh"]),
    ("In which country is the Great Wall located?", "China", ["Japan", "India", "Mongolia"]),
    ("Which is the longest river in Africa?", "Nile", ["Congo", "Niger", "Zambezi"]),
    ("Which is the largest ocean on Earth?", "Pacific", ["Atlantic", "Indian", "Arctic"]),
    ("What is the tallest mountain above sea level?", "Mount Everest", ["K2", "Kangchenjunga", "Denali"]),
    ("Which country is the largest by land area?", "Russia", ["Canada", "China", "United States"]),
    ("In which city is the Eiffel Tower located?", "Paris", ["Rome", "Berlin", "Madrid"]),
    ("Which language has the most native speakers?", "Mandarin Chinese", ["English", "Spanish", "Hindi"]),
    ("Who developed the theory of evolution by natural selection?", "Charles Darwin", ["Gregor Mendel", "Louis Pasteur", "Carl Linnaeus"]),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", ["Buzz Aldrin", "Yuri Gagarin", "Michael Collins"]),
    ("What does HTTP stand for?", "HyperText Transfer Protocol", ["HyperText Transmission Process", "High Transfer Text Protocol", "Host Transfer Text Program"]),
    ("How many bits are in a byte?", "8", ["4", "16", "32"]),
    ("Who created the Linux kernel?", "Linus Torvalds", ["Richard Stallman", "Ken Thompson", "Dennis Ritchie"]),
    ("What does CPU stand for?", "Central Processing Unit", ["Computer Personal Unit", "Central Program Utility", "Core Processing Unit"]),
    ("In which year did World War II end?", "1945", ["1918", "1939", "1951"]),
    ("Which ancient civilization built the pyramids at Giza?", "Ancient Egyptians", ["Ancient Greeks", "Ancient Romans", "Ancient Persians"]),
    ("What is the currency of Japan?", "Yen", ["Won", "Yuan", "Baht"]),
    ("On which continent is the Sahara Desert?", "Africa", ["Asia", "Australia", "South America"]),
    ("How many sides does a hexagon have?", "6", ["5", "7", "8"]),
]


# ----------------------------------------------------------------------------------- arithmetic
def _arith_items(rng: random.Random, n: int) -> list[dict]:
    templates = []

    def t1(r):
        a, b, c = r.randint(20, 90), r.randint(10, 60), r.randint(5, 40)
        if a + b - c < 0:
            return None
        q = (f"A shop had {a} apples. It received {b} more, then sold {c}. "
             f"How many apples are left?")
        return q, a + b - c

    def t2(r):
        a, b, c = r.randint(3, 12), r.randint(4, 15), r.randint(2, 30)
        if a * b - c < 0:
            return None
        q = (f"There are {a} boxes with {b} pencils each. {c} pencils are removed. "
             f"How many pencils remain?")
        return q, a * b - c

    def t3(r):
        a, b, c = r.randint(20, 90), r.randint(2, 9), r.randint(5, 60)
        q = (f"A train travels {a} km per hour for {b} hours, then {c} km more. "
             f"How many km did it travel in total?")
        return q, a * b + c

    def t4(r):
        a, b, c = r.randint(100, 300), r.randint(20, 100), r.randint(5, 80)
        if b + c > a:
            return None
        q = (f"A tank holds {a} litres. It already contains {b} litres, and {c} litres are "
             f"added. How many litres are in the tank now?")
        return q, b + c

    def t5(r):
        a, b, c = r.randint(30, 90), r.randint(5, 40), r.randint(5, 40)
        if a - b + c < 0:
            return None
        q = (f"Sam has {a} marbles. He gives {b} away, then wins {c}. "
             f"How many marbles does he have now?")
        return q, a - b + c

    templates = [t1, t2, t3, t4, t5]
    items, seen_q = [], set()
    guard = 0
    while len(items) < n and guard < n * 100:
        guard += 1
        tmpl = templates[len(items) % len(templates)]
        got = tmpl(rng)
        if not got:
            continue
        q, ans = got
        if q in seen_q:
            continue
        seen_q.add(q)
        pool = {ans - 2, ans - 1, ans + 1, ans + 2, ans + 10, ans - 10}
        pool = {v for v in pool if v >= 0 and v != ans}
        distr = rng.sample(sorted(pool), 3)
        choices = [str(ans)] + [str(d) for d in distr]
        rng.shuffle(choices)
        items.append({
            "id": f"arith-{len(items)+1:03d}",
            "task": "arithmetic",
            "question": q,
            "choices": choices,
            "answer": choices.index(str(ans)),
        })
    return items


def build() -> list[dict]:
    rng = random.Random(SEED)
    items = _arith_items(rng, 100)
    for i, (q, correct, distractors) in enumerate(FACTS, start=1):
        choices = [correct] + list(distractors)
        rng.shuffle(choices)
        items.append({
            "id": f"know-{i:03d}",
            "task": "knowledge",
            "question": q,
            "choices": choices,
            "answer": choices.index(correct),
        })
    return items


def render(items: list[dict]) -> str:
    return "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n"


def _display(path: Path) -> str:
    """Repo-relative in output so committed transcripts never contain absolute home paths."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="determinism check (and file sync if present)")
    args = ap.parse_args()
    text = render(build())
    if args.check:
        if text != render(build()):
            print("NON-DETERMINISTIC build", file=sys.stderr)
            return 1
        if OUT.exists() and OUT.read_text(encoding="utf-8") != text:
            print(f"STALE {_display(OUT)}: regenerate with python3 tools/make_benchmark.py", file=sys.stderr)
            return 1
        state = "in sync" if OUT.exists() else "generated on demand"
        print(f"OK {_display(OUT)} ({len(text.splitlines())} items, {state})")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {_display(OUT)} ({len(text.splitlines())} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
