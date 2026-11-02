"""Training data for the transaction-text model.

There is no public labelled dataset of Indian personal-expense descriptions, so the model is trained
on hand-written *seed phrases* (English + romanised Hinglish) that are expanded with light, label-
preserving augmentation (prefixes, suffixes, typos, casing, dropped words).

IMPORTANT for honest evaluation: every augmented row keeps the ``group`` id of its seed, so
cross-validation splits by seed (a phrase and its variants never appear on both sides). A separate,
independently written ``data/handwritten_test_set.csv`` is used as the final check. Both are still
synthetic-style data - see the README "Limitations" section.

Labels
------
category : one of config.CATEGORIES
context  : essential | discretionary | mixed
           (the *likely* nature of the spend; a statistical guess, not a judgement of the user)
"""
from __future__ import annotations

import random
import re

import pandas as pd

from . import config

CONTEXTS = ["essential", "discretionary", "mixed"]

_S = lambda block: [ln.strip() for ln in block.strip().splitlines() if ln.strip()]  # noqa: E731

SEEDS: dict[tuple[str, str], list[str]] = {
    # ------------------------------------------------------------------ Food
    ("Food", "essential"): _S("""
weekly groceries
vegetables and fruits from sabzi mandi
milk and curd
atta rice dal monthly ration
kirana store saaman
mess fees for the month
tiffin service monthly payment
cooking oil and spices
eggs and bread
grocery shopping at dmart
bigbasket order for vegetables
ghar ke liye sabzi
doodh ka bill
fruits for the week
paneer and vegetables for dinner
rashan ka saaman
tea powder sugar and biscuits
hostel mess charges
"""),
    ("Food", "discretionary"): _S("""
late night swiggy order
zomato order burger and fries
dinner with friends at restaurant
dost ke saath restaurant mein dinner
starbucks coffee
pizza party
ice cream and dessert
cafe hangout with friends
ordered biryani online
midnight maggi and snacks delivery
buffet lunch on weekend
momos and cold drink
birthday party dinner treat
bubble tea
chocolates and chips binge
dominos large pizza
mcdonalds meal
brunch at new cafe
party snacks and drinks
kfc bucket
cappuccino and brownie at cafe
blinkit chips and ice cream
"""),
    ("Food", "mixed"): _S("""
lunch with colleagues
chai and samosa
canteen lunch
tea at tapri
breakfast at udupi
dosa for lunch
thali meal
juice from stall
lunch outside today
snacks from vending machine
coffee with classmate
street food chaat
birthday cake
office cafeteria lunch
"""),
    # ---------------------------------------------------------------- Travel
    ("Travel", "essential"): _S("""
monthly bus pass
metro card recharge
daily commute auto fare
petrol for bike to college
local train season ticket
bus fare to college
college jaane ke liye metro recharge
petrol for office commute
shared auto to office
train ticket home for exam
parking charges at office
daily bus ticket
"""),
    ("Travel", "discretionary"): _S("""
weekend trip to goa
hotel booking for vacation
flight tickets for holiday
cab for night out with friends
road trip with friends petrol and tolls
resort stay
manali trip bus tickets
bike rental on vacation
airbnb for long weekend
sightseeing tour tickets
uber to club
trekking trip transport
dost ke saath ghoomne gaye
"""),
    ("Travel", "mixed"): _S("""
cab to airport
ola ride
uber ride home
petrol
train ticket
auto fare
rapido bike taxi
toll charges
flight ticket to hometown
bus ticket to hometown for festival
fuel top up
rickshaw
"""),
    # -------------------------------------------------------------- Shopping
    ("Shopping", "essential"): _S("""
replacement shoes since old ones tore
underwear and socks
new phone charger as old one stopped working
raincoat for monsoon
slippers
basic t shirts and innerwear
school uniform
laptop bag for college
mobile screen guard replacement
formal shirt for interview
woollen clothes for winter
purana chappal toot gaya naya liya
"""),
    ("Shopping", "discretionary"): _S("""
amazon impulse buy
new sneakers during sale
myntra clothes haul
bought headphones i did not need
makeup and skincare products
flipkart sale deals
smartwatch
fancy sunglasses
gaming accessories
perfume
trendy tops from online sale
extra pair of jeans
branded watch
phone cover and cute accessories
shopping at mall with friends
bina zaroorat ke online shopping
"""),
    ("Shopping", "mixed"): _S("""
amazon order
gift for friends birthday
birthday gift for mom
clothes for wedding
new shoes
flipkart order
kurta for festival
bag
diwali shopping
wedding gift
jeans
backpack
kapde kharide
"""),
    # ------------------------------------------------------------- Education
    ("Education", "essential"): _S("""
college fees
semester exam fee
textbooks for the semester
college textbooks purchase
stationery for class
project report printing
lab manual and record notebook
tuition fees
photocopy of notes
exam form fee
library membership
gate exam registration fee
thesis binding
padhai ke liye notebook aur pen
"""),
    ("Education", "discretionary"): _S("""
novel for leisure reading
bought udemy course on sale which i may not finish
extra online course subscription
comic books
magazine subscription
kindle books for fun
hobby painting class
guitar lessons
skill workshop impulse purchase
storybooks for timepass
"""),
    ("Education", "mixed"): _S("""
online course
coding bootcamp
coaching class fees
certification exam fee
coursera subscription
interview preparation material
leetcode premium
english speaking course
reference books
ebooks for placement preparation
"""),
    # ----------------------------------------------------------------- Bills
    ("Bills", "essential"): _S("""
electricity bill
mobile recharge
wifi bill
monthly rent
water bill
gas cylinder refill
insurance premium
credit card bill payment
bijli ka bill
room rent to landlord
broadband recharge
society maintenance charges
loan emi
postpaid phone bill
lic premium
property tax
phone recharge for the month
"""),
    ("Bills", "mixed"): _S("""
dth recharge
upgraded mobile plan with extra data
late payment fee
bank annual charges
cloud storage subscription
domain renewal
vpn subscription
newspaper bill
cable tv bill
app subscription renewal
"""),
    # --------------------------------------------------------- Entertainment
    ("Entertainment", "discretionary"): _S("""
movie tickets
netflix subscription
spotify premium
gaming top up
bgmi uc purchase
concert tickets
bowling with friends
ott subscription
cricket match tickets
amusement park entry
pc game on steam
pvr popcorn combo
unplanned subscription
standup comedy show
youtube premium
arcade zone
escape room
karaoke night
ps5 game
disney hotstar
beer with friends
film dekhne gaye
"""),
    ("Entertainment", "mixed"): _S("""
turf booking for football with friends
book club membership
annual hotstar plan
board games
museum entry
podcast app subscription
badminton court booking
cricket net practice fee
"""),
    # ---------------------------------------------------------------- Health
    ("Health", "essential"): _S("""
doctor consultation fees
medicines from pharmacy
blood test at lab
dental checkup
eye checkup
prescription glasses
vaccination
fever tablets
dawai ka kharcha
hospital bill
x ray scan
physiotherapy session
bp medicines for dad
clinic visit
"""),
    ("Health", "mixed"): _S("""
gym membership
protein powder
yoga class
vitamins and supplements
multivitamin tablets
health checkup package
zumba classes
sports nutrition
swimming pool monthly pass
"""),
    ("Health", "discretionary"): _S("""
spa and massage
facial and skin treatment
cosmetic clinic session
luxury salon spa day
expensive protein bars
wellness retreat
detox juice cleanse
"""),
    # ------------------------------------------------------------- Household
    ("Household", "essential"): _S("""
cleaning supplies
plumber for leaking tap
electrician charges
kitchen utensils
light bulbs and extension board
maid salary
detergent and dishwash
toilet cleaner and phenyl
lpg stove repair
water purifier service
gas stove lighter
bucket and mug
ghar ka saaman
bathroom fittings repair
mosquito repellent
pest control
broom and mop
water can refill
kaamwali bai ka paisa
"""),
    ("Household", "mixed"): _S("""
mattress
table fan
furniture for room
curtains
storage boxes
iron box
study table
cooler for summer
bedsheets and pillows
room heater
"""),
    ("Household", "discretionary"): _S("""
home decor items
fairy lights for room
indoor plants
fancy cushions
wall posters and frames
scented candles
aesthetic lamp
showpiece
diwali decoration items
"""),
    # ----------------------------------------------------------------- Other
    ("Other", "mixed"): _S("""
donation to temple
haircut
miscellaneous
tip to delivery person
atm withdrawal fee
lent money to friend
random expense
charity donation
salon haircut and shave
stamp and envelope
"""),
    ("Other", "essential"): _S("""
laundry and ironing
courier charges for documents
passport application fee
notary and stamp paper
aadhar update fee
photocopy and scanning of id
passport photos
document attestation
driving licence renewal
bank charges
"""),
    ("Other", "discretionary"): _S("""
pet toys and treats
lottery ticket
astrology consultation
lucky charm
cigarettes and paan
bet with friends
"""),
}

_PREFIXES = ["", "", "", "", "paid for ", "bought ", "spent on ", "aaj ", "today ", "yesterday ", "paid "]
_SUFFIXES = ["", "", "", "", " today", " yesterday", " online", " upi", " cash", " this week",
             " paid by card", " rs {amt}", " ₹{amt}"]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["delete", "swap", "double"])
    if kind == "delete":
        return word[:i] + word[i + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def augment(seed_text: str, n: int, rng: random.Random) -> list[str]:
    """The seed itself plus up to ``n-1`` distinct, label-preserving variants."""
    out = [seed_text]
    seen = {seed_text.lower()}
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        words = seed_text.split()
        if len(words) >= 3 and rng.random() < 0.10:
            words.pop(rng.randrange(len(words)))
        if rng.random() < 0.15:
            j = rng.randrange(len(words))
            words[j] = _typo(words[j], rng)
        text = rng.choice(_PREFIXES) + " ".join(words) + rng.choice(_SUFFIXES).format(amt=rng.choice([50, 120, 250, 499, 1200]))
        text = rng.choice([str.lower, str.capitalize, str.title, str.lower])(text).strip()
        if text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def build_training_frame(n_aug: int = 12, seed: int = 0) -> pd.DataFrame:
    """Columns: text, category, context, group (seed id), is_seed."""
    rng = random.Random(seed)
    rows = []
    gid = 0
    for (cat, ctx), phrases in SEEDS.items():
        for phrase in phrases:
            for k, text in enumerate(augment(phrase, n_aug, rng)):
                rows.append({"text": text, "category": cat, "context": ctx, "group": gid, "is_seed": k == 0})
            gid += 1
    return pd.DataFrame(rows)


def seed_frame() -> pd.DataFrame:
    """Just the un-augmented seed phrases."""
    return pd.DataFrame([{"text": p, "category": c, "context": x}
                         for (c, x), ps in SEEDS.items() for p in ps])


def load_handwritten_test() -> pd.DataFrame:
    return pd.read_csv(config.DATA_DIR / "handwritten_test_set.csv")


def normalise_for_overlap(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def validate_seeds() -> None:
    """Sanity checks: valid labels and no phrase listed under two different labels."""
    seen: dict[str, tuple[str, str]] = {}
    for (cat, ctx), phrases in SEEDS.items():
        assert cat in config.CATEGORIES, cat
        assert ctx in CONTEXTS, ctx
        for p in phrases:
            key = normalise_for_overlap(p)
            assert key not in seen, f"duplicate seed phrase: {p!r} ({seen[key]} vs {(cat, ctx)})"
            seen[key] = (cat, ctx)
