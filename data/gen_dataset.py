"""Generate the Kharcha dataset: Indian bank/UPI SMS + Hinglish voice notes with
program-generated ground truth. Labels come from the template parameters, never
from a model, so they are exact.

Usage:
  python data/gen_dataset.py                 # 1200 train / 150 val / 150 test
  python data/gen_dataset.py --train 2000    # bigger
  python data/gen_dataset.py --augment       # extra Tinker-sampled paraphrases (needs TINKER_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import HINTS_HEADER, build_system_prompt, to_json, validate  # noqa: E402

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- merchants
# (display name, vpa, category). Display is how it appears in app notifications;
# bank SMS upper-case it and sometimes truncate.
MERCHANTS = [
    # groceries
    ("Sharma Kirana Store", "sharmakirana@ybl", "groceries"),
    ("Gupta General Store", "guptageneral@okaxis", "groceries"),
    ("BigBasket", "bigbasket@hdfcbank", "groceries"),
    ("Blinkit", "blinkit@ybl", "groceries"),
    ("Zepto", "zepto@axl", "groceries"),
    ("DMart", "dmart@icici", "groceries"),
    ("Reliance Fresh", "reliancefresh@ybl", "groceries"),
    ("Mother Dairy Booth", "motherdairy@paytm", "groceries"),
    ("Agarwal Sabzi Bhandar", "agarwalsabzi@ybl", "groceries"),
    ("Nature Basket", "naturesbasket@icici", "groceries"),
    # food_delivery
    ("Zomato", "zomato@hdfcbank", "food_delivery"),
    ("Swiggy", "swiggy@icici", "food_delivery"),
    ("Dominos Pizza", "dominos@ybl", "food_delivery"),
    ("Haldirams", "haldirams@paytm", "food_delivery"),
    ("Bikanervala", "bikanervala@okicici", "food_delivery"),
    ("Chai Point", "chaipoint@ybl", "food_delivery"),
    ("Burger King", "burgerking@axl", "food_delivery"),
    ("Theobroma", "theobroma@ybl", "food_delivery"),
    # transport
    ("Uber India", "uber@hdfcbank", "transport"),
    ("Ola Cabs", "ola@icici", "transport"),
    ("Rapido", "rapido@ybl", "transport"),
    ("IRCTC", "irctc@sbi", "transport"),
    ("Delhi Metro Rail", "dmrc@ybl", "transport"),
    ("Indian Oil Petrol Pump", "iocl@paytm", "transport"),
    ("HP Petrol Pump", "hpcl@okaxis", "transport"),
    ("FASTag Recharge", "fastag@icici", "transport"),
    ("RedBus", "redbus@ybl", "transport"),
    # utilities
    ("BSES Rajdhani Power", "bses@ybl", "utilities"),
    ("Tata Power DDL", "tatapower@icici", "utilities"),
    ("Airtel Prepaid", "airtel@ybl", "utilities"),
    ("Jio Recharge", "jio@paytm", "utilities"),
    ("Indane Gas Booking", "indane@sbi", "utilities"),
    ("Delhi Jal Board", "djb@ybl", "utilities"),
    ("Hathway Broadband", "hathway@okaxis", "utilities"),
    ("Vodafone Idea", "vi@ybl", "utilities"),
    # medical
    ("Apollo Pharmacy", "apollopharmacy@ybl", "medical"),
    ("Tata 1mg", "1mg@icici", "medical"),
    ("Dr Mehta Clinic", "drmehta@okaxis", "medical"),
    ("Max Hospital Saket", "maxhealthcare@hdfcbank", "medical"),
    ("Dr Lal PathLabs", "lalpathlabs@ybl", "medical"),
    ("Wellness Forever", "wellnessforever@paytm", "medical"),
    ("Fortis Hospital", "fortis@icici", "medical"),
    # rent_emi
    ("HDFC Home Loan EMI", None, "rent_emi"),
    ("Bajaj Finance EMI", None, "rent_emi"),
    ("SBI Car Loan EMI", None, "rent_emi"),
    ("Mr Suresh Verma Rent", "sureshverma@okhdfcbank", "rent_emi"),
    ("Mrs Kavita Landlord Rent", "kavita.rent@ybl", "rent_emi"),
    # shopping
    ("Amazon", "amazon@apl", "shopping"),
    ("Flipkart", "flipkart@axl", "shopping"),
    ("Myntra", "myntra@ybl", "shopping"),
    ("Reliance Trends", "trends@icici", "shopping"),
    ("Croma", "croma@hdfcbank", "shopping"),
    ("Nykaa", "nykaa@ybl", "shopping"),
    ("Decathlon", "decathlon@icici", "shopping"),
    ("Westside", "westside@ybl", "shopping"),
    # subscriptions
    ("Netflix", "netflix@hdfcbank", "subscriptions"),
    ("Disney Hotstar", "hotstar@icici", "subscriptions"),
    ("Spotify", "spotify@ybl", "subscriptions"),
    ("Amazon Prime", "amazonprime@apl", "subscriptions"),
    ("YouTube Premium", "google@okaxis", "subscriptions"),
    ("Times Prime", "timesprime@ybl", "subscriptions"),
    # other (P2P to non-family, misc)
    ("Ramesh Electrician", "ramesh.elec@ybl", "other"),
    ("Sunil Plumber", "sunil.plumber@paytm", "other"),
    ("Priya Sharma", "priya.sharma@okaxis", "other"),
    ("Amit Kumar", "amitk@ybl", "other"),
    ("Society Maintenance", "rwa.maint@icici", "other"),
    ("LIC Premium", "lic@sbi", "other"),
    ("Paytm Wallet", "paytmwallet@paytm", "other"),
    ("Neha Gupta", "nehag@ybl", "other"),
]

# Family: used only in voice notes with a relation word, or via hints.
FAMILY = [
    ("Rahul", "beta", "rahulk@ybl"),
    ("Pooja", "beti", "pooja.s@okaxis"),
    ("Mummy", "mummy", "sunita.d@ybl"),
    ("Papa", "papa", "rk.devi@sbi"),
    ("Bhaiya", "bhaiya", "vikas.k@ybl"),
    ("Didi", "didi", "anjali.m@paytm"),
    ("Chachu", "chachu", "mohan.k@icici"),
]

FAMILY_FULL_NAMES = {
    "beta": "RAHUL KUMAR", "beti": "POOJA SHARMA", "mummy": "SUNITA DEVI", "papa": "R K SHARMA",
    "bhaiya": "VIKAS KUMAR", "didi": "ANJALI MEHRA", "chachu": "MOHAN KUMAR",
}

EMPLOYERS = ["ACME TECH PVT LTD", "INFOSYS LTD", "TATA CONSULTANCY SERVICES", "URBAN COMPANY", "HCL TECHNOLOGIES", "WIPRO LTD", "KOTAK LIFE INSURANCE", "DELHI PUBLIC SCHOOL"]
ATM_LOCATIONS = ["CP DELHI", "SECTOR 18 NOIDA", "ANDHERI WEST", "KORAMANGALA", "SALT LAKE", "MG ROAD GURGAON", "LAJPAT NAGAR", "BANER PUNE"]
BANKS = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK", "PNB"]

AMOUNT_RANGES = {
    "groceries": (40, 4500),
    "food_delivery": (90, 2200),
    "transport": (25, 3500),
    "utilities": (99, 6000),
    "medical": (60, 25000),
    "rent_emi": (4500, 45000),
    "shopping": (199, 35000),
    "subscriptions": (99, 1499),
    "other": (50, 20000),
    "family_transfer": (200, 25000),
    "salary_income": (18000, 180000),
    "cash_withdrawal": (500, 20000),
}

# --------------------------------------------------------------------------- helpers
def rupees(amount: float, style: str) -> str:
    """Render amount in common Indian SMS styles."""
    if style == "plain2":
        return f"{amount:.2f}"
    if style == "plain":
        return f"{amount:.2f}".rstrip("0").rstrip(".") if amount != int(amount) else str(int(amount))
    if style == "indian":
        return indian_group(amount, decimals=2)
    if style == "indian0":
        return indian_group(amount, decimals=0)
    return f"{amount:.2f}"


def indian_group(amount: float, decimals: int) -> str:
    whole = int(amount)
    frac = amount - whole
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    if decimals:
        return f"{s}.{int(round(frac * 100)):02d}"
    return s


def fmt_date(d: date, style: str) -> str:
    return {
        "dmy_slash2": d.strftime("%d/%m/%y"),
        "dmy_dash2": d.strftime("%d-%m-%y"),
        "dmy_dash4": d.strftime("%d-%m-%Y"),
        "dmy_slash4": d.strftime("%d/%m/%Y"),
        "dMony2": d.strftime("%d%b%y"),
        "d-Mon-y2": d.strftime("%d-%b-%y"),
        "d-Mon-y4": d.strftime("%d-%b-%Y"),
        "dMon": d.strftime("%d %b"),
    }[style]


def ref(n=12) -> str:
    return "".join(random.choice("0123456789") for _ in range(n))


def last4() -> str:
    return f"{random.randint(0, 9999):04d}"


def pick_amount(category: str) -> float:
    lo, hi = AMOUNT_RANGES[category]
    if category in ("subscriptions",):
        return float(random.choice([99, 149, 199, 299, 499, 649, 799, 1499]))
    if category == "cash_withdrawal":
        return float(random.choice([500, 1000, 2000, 3000, 5000, 10000, 20000]))
    if category in ("rent_emi", "salary_income"):
        return float(random.randint(lo // 100, hi // 100) * 100)
    if random.random() < 0.55:
        return float(random.randint(lo, hi))
    return round(random.uniform(lo, hi), 2)


def upper_trunc(name: str, maxlen: int) -> str:
    return name.upper()[:maxlen].strip()


@dataclass
class Example:
    text: str
    label: dict
    kind: str  # template id, for coverage checks
    hints: list[str] | None = None


# --------------------------------------------------------------------------- SMS templates
def sms_merchant_debit(d: date, merchant, l4: str, with_hint: bool = False) -> Example:
    name, vpa, cat = merchant
    amt = pick_amount(cat)
    hints = None
    if with_hint:
        new_cat = random.choice([c for c in AMOUNT_RANGES if c not in (cat, "salary_income", "cash_withdrawal")])
        hints = [f"Treat {name} as {new_cat}."]
        cat = new_cat
    bank = random.choice(BANKS)
    r = ref()
    t = random.choice(["hdfc_upi", "hdfc_card", "sbi_upi", "icici_upi", "axis_upi", "kotak_upi", "pnb_upi", "autopay", "icici_card", "sbi_card"])
    channel = "upi"
    cp = upper_trunc(name, 28)
    if t == "hdfc_upi":
        text = f"Sent Rs.{rupees(amt,'plain2')} From HDFC Bank A/C *{l4} To {cp} On {fmt_date(d,'dmy_slash2')} Ref {r} Not You? Call 18002586161/SMS BLOCK UPI to 7308080808"
        bank = "HDFC"
    elif t == "hdfc_card":
        channel = "card"
        text = f"Thanks for using HDFC Bank Card x{l4} for Rs {rupees(amt,'plain2')} at {cp} on {fmt_date(d,'dmy_dash4')} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}. Avl bal: Rs {indian_group(random.uniform(2000,90000),2)}"
    elif t == "sbi_upi":
        cp = upper_trunc(name, 18)
        text = f"Dear UPI user A/C X{l4} debited by {rupees(amt,'plain')} on date {fmt_date(d,'dMony2')} trf to {cp} Refno {r}. If not u? call 1800111109. -SBI"
    elif t == "icici_upi":
        text = f"ICICI Bank Acct XX{l4[1:]} debited for Rs {rupees(amt,'plain2')} on {fmt_date(d,'d-Mon-y2')}; {cp} credited. UPI:{r}. Call 18002662 for dispute. SMS BLOCK {l4[1:]} to 9215676766."
        l4_out = l4  # 3 digits shown; label last4 unknown -> keep None
        return Example(text, _label("debit", amt, cp, channel, None, d, cat), t, hints)
    elif t == "axis_upi":
        text = f"INR {rupees(amt,'plain2')} debited A/c no. XX{l4} {fmt_date(d,'dmy_dash2')} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d} UPI/P2M/{r}/{cp} Not you? SMS BLOCKUPI Cust ID to 919951860002 - Axis Bank"
    elif t == "kotak_upi":
        cp = vpa or cp
        text = f"Sent Rs.{rupees(amt,'plain2')} from Kotak Bank AC X{l4} to {cp} on {fmt_date(d,'dmy_dash2')}.UPI Ref {r}. Not you, kotak.com/fraud"
    elif t == "pnb_upi":
        cp = vpa or cp
        text = f"Your A/c XX{l4} debited Rs. {rupees(amt,'plain2')} on {fmt_date(d,'dmy_dash4')} to VPA {cp}. UPI Ref {r}. Not you? Call 18001802222 -PNB"
    elif t == "autopay":
        text = f"Rs {rupees(amt,'plain2')} debited from {bank} Bank A/c XX{l4} for {cp} via UPI Autopay on {fmt_date(d,'dmy_dash2')}. UPI Ref {r}. To cancel mandate visit your UPI app."
    elif t == "icici_card":
        channel = "card"
        text = f"INR {rupees(amt,'indian')} spent on ICICI Bank Card XX{l4} on {fmt_date(d,'d-Mon-y2')} at {cp}. Avl Limit: INR {indian_group(random.uniform(10000,200000),2)}. If not you, call 18002662."
    else:  # sbi_card
        channel = "card"
        text = f"Rs.{rupees(amt,'plain2')} spent on your SBI Credit Card ending {l4} at {cp} on {fmt_date(d,'dmy_slash2')}. Trxn. not done by you? Report at https://sbicard.com/Dispute"
    return Example(text, _label("debit", amt, cp, channel, l4, d, cat), t, hints)


def sms_app_notification(d: date, merchant, l4: str) -> Example:
    name, vpa, cat = merchant
    amt = pick_amount(cat)
    r = ref()
    t = random.choice(["phonepe", "gpay", "paytm", "cred"])
    bank = random.choice(BANKS)
    if t == "phonepe":
        text = f"₹{rupees(amt,'indian0' if amt==int(amt) else 'indian')} paid to {name} · Debited from {bank} Bank ****{l4} · Txn ID T{d.strftime('%y%m%d')}{ref(10)}"
    elif t == "gpay":
        text = f"You paid ₹{rupees(amt,'plain')} to {name} using Google Pay. UPI transaction ID {r}"
    elif t == "paytm":
        text = f"Paid ₹{rupees(amt,'plain')} to {name} from Paytm UPI linked {bank} Bank ****{l4}. Order ID {ref(8)}"
    else:
        text = f"Payment of ₹{rupees(amt,'indian0' if amt==int(amt) else 'indian')} to {name} successful via CRED UPI from {bank} Bank XX{l4}. Ref {r}"
    return Example(text, _label("debit", amt, name, "upi", l4 if t != "gpay" else None, d if t == "phonepe" else None, cat), t)


def sms_salary(d: date, l4: str) -> Example:
    amt = pick_amount("salary_income")
    emp = random.choice(EMPLOYERS)
    bank = random.choice(BANKS)
    t = random.choice(["neft_salary", "imps_credit_salary"])
    if t == "neft_salary":
        text = f"Your A/C XXXXX{l4} Credited INR {indian_group(amt,2)} on {fmt_date(d,'dmy_slash2')} by NEFT from {emp} (Ref {ref(16)}). Avl Bal INR {indian_group(amt + random.uniform(1000,50000),2)} -{bank}"
        ch = "neft"
    else:
        text = f"Rs {indian_group(amt,2)} credited to A/c XX{l4} on {fmt_date(d,'dmy_dash2')} by {emp} SALARY {d.strftime('%b').upper()} via IMPS Ref {ref()}. -{bank} Bank"
        ch = "imps"
    return Example(text, _label("credit", amt, emp, ch, l4, d, "salary_income"), t)


def sms_atm(d: date, l4: str) -> Example:
    amt = pick_amount("cash_withdrawal")
    bank = random.choice(BANKS)
    loc = random.choice(ATM_LOCATIONS)
    t = random.choice(["atm_a", "atm_b"])
    if t == "atm_a":
        text = f"Rs {rupees(amt,'plain2')} withdrawn from {bank} Bank ATM at {loc} from A/c x{l4} on {fmt_date(d,'dmy_dash2')}. Avl Bal Rs {indian_group(random.uniform(1000,80000),2)}"
    else:
        text = f"Dear Customer, INR {indian_group(amt,2)} has been debited from A/c XX{l4} for ATM WDL at {loc} on {fmt_date(d,'d-Mon-y4')}. Not you? Call {bank} immediately."
    return Example(text, _label("debit", amt, f"{bank} Bank ATM {loc}" if t == "atm_a" else f"ATM WDL {loc}", "atm", l4, d, "cash_withdrawal"), t)


def sms_emi(d: date, l4: str) -> Example:
    name = random.choice([m for m in MERCHANTS if m[2] == "rent_emi" and m[1] is None])[0]
    amt = pick_amount("rent_emi")
    bank = random.choice(BANKS)
    text = random.choice([
        f"Rs {indian_group(amt,2)} debited from A/c XX{l4} towards {name} on {fmt_date(d,'dmy_dash2')}. Ref {ref(10)}. -{bank} Bank",
        f"Dear Customer, EMI of INR {indian_group(amt,2)} for {name} has been successfully debited from your A/c ending {l4} on {fmt_date(d,'d-Mon-y2')}.",
        f"ECS/NACH debit of Rs.{rupees(amt,'plain2')} for {name} from A/c X{l4} on {fmt_date(d,'dmy_slash4')} processed. -{bank}",
    ])
    return Example(text, _label("debit", amt, name, "unknown", l4, d, "rent_emi"), "emi")


def sms_rent_transfer(d: date, l4: str) -> Example:
    m = random.choice([m for m in MERCHANTS if m[2] == "rent_emi" and m[1] is not None])
    name, vpa, _ = m
    person = name.replace(" Rent", "")
    amt = pick_amount("rent_emi")
    bank = random.choice(BANKS)
    text = random.choice([
        f"Rs {indian_group(amt,2)} debited from A/c XX{l4} on {fmt_date(d,'dmy_dash2')} to {upper_trunc(person,30)} (IMPS Ref {ref()}) Remarks: RENT {d.strftime('%b').upper()}. -{bank}",
        f"Sent Rs.{rupees(amt,'plain2')} From {bank} Bank A/C *{l4} To {upper_trunc(person,28)} On {fmt_date(d,'dmy_slash2')} Ref {ref()} Note: house rent",
    ])
    ch = "imps" if "IMPS" in text else "upi"
    return Example(text, _label("debit", amt, upper_trunc(person, 30) if "IMPS" in text else upper_trunc(person, 28), ch, l4, d, "rent_emi"), "rent_transfer")


def sms_p2p(d: date, l4: str, with_hint: bool) -> Example:
    """Person-to-person transfer. Family only when hint says so (or never, in SMS)."""
    if with_hint:
        fam = random.choice(FAMILY)
        full = FAMILY_FULL_NAMES[fam[1]]
        cat = "family_transfer"
        hints = [f"{full.title()} is family ({fam[1]}); transfers to them are family_transfer."]
    else:
        full = random.choice(["ROHIT SAINI", "MEENA KUMARI", "ARJUN NAIR", "FARHAN KHAN", "DEEPAK YADAV", "SIMRAN KAUR", "AMIT KUMAR", "PRIYA SHARMA"])
        cat = "other"
        hints = None
    amt = pick_amount("family_transfer" if cat == "family_transfer" else "other")
    bank = random.choice(BANKS)
    t = random.choice(["p2p_upi", "p2p_imps", "p2p_credit"])
    if t == "p2p_upi":
        text = f"Sent Rs.{rupees(amt,'plain2')} From {bank} Bank A/C *{l4} To {full} On {fmt_date(d,'dmy_slash2')} Ref {ref()} Not You? Call 18002586161"
        lab = _label("debit", amt, full, "upi", l4, d, cat)
    elif t == "p2p_imps":
        text = f"Rs {indian_group(amt,2)} debited from A/c XX{l4} on {fmt_date(d,'dmy_dash2')} to {full} (IMPS Ref {ref()}). -{bank}"
        lab = _label("debit", amt, full, "imps", l4, d, cat)
    else:
        text = f"Rs {indian_group(amt,2)} credited to A/c XX{l4} on {fmt_date(d,'dmy_dash2')} from {full} via UPI Ref {ref()}. Avl Bal Rs {indian_group(random.uniform(5000,90000),2)} -{bank}"
        lab = _label("credit", amt, full, "upi", l4, d, cat)
    return Example(text, lab, t, hints)


# --------------------------------------------------------------------------- voice notes (Hinglish)
VOICE_MERCHANT_WORDS = {
    "groceries": ["sabzi wale", "doodh wale", "kirane", "Sharma ji ki dukaan", "ration", "fal wale", "Blinkit", "BigBasket"],
    "food_delivery": ["Zomato", "Swiggy", "chaat", "dhabe", "pizza", "momos wale", "chai nashta"],
    "transport": ["auto", "rickshaw", "Uber", "Ola", "metro card", "petrol", "bus", "Rapido"],
    "utilities": ["bijli ka bill", "paani ka bill", "gas cylinder", "Airtel recharge", "Jio recharge", "wifi ka bill", "DTH recharge"],
    "medical": ["dawai", "doctor", "Apollo", "blood test", "hospital", "chashma", "physio"],
    "rent_emi": ["kiraya", "ghar ka rent", "EMI", "makaan malik"],
    "shopping": ["kapde", "Amazon", "Flipkart", "jooti", "sari", "mixer", "gift"],
    "subscriptions": ["Netflix", "Hotstar", "Spotify", "Prime", "YouTube"],
    "other": ["electrician", "plumber", "maid", "society maintenance", "mandir daan", "shaadi shagun", "LIC"],
    "cash_withdrawal": ["ATM se cash", "ATM", "bank se cash"],
}

VOICE_AMOUNT_WORDS = {
    50: ["pachas", "50"], 60: ["saath", "60"], 80: ["assi", "80"], 100: ["sau", "100", "ek sau"], 150: ["dedh sau", "150"],
    200: ["do sau", "200"], 250: ["dhai sau", "250"], 300: ["teen sau", "300"], 500: ["paanch sau", "500", "500 rupaye"],
    1000: ["ek hazaar", "1000", "hazaar"], 1200: ["baarah sau", "1200"], 1500: ["pandrah sau", "1500", "dedh hazaar"],
    2000: ["do hazaar", "2000"], 2500: ["dhai hazaar", "2500"], 3000: ["teen hazaar", "3000"], 5000: ["paanch hazaar", "5000"],
    10000: ["das hazaar", "10000", "10 hazaar"], 15000: ["pandrah hazaar", "15000"], 20000: ["bees hazaar", "20000"],
}

DAY_WORDS = {"aaj": 0, "kal": -1, "parso": -2}


def voice_note(d: date, with_hint: bool) -> Example:
    """Short Hinglish spoken note. Date is relative to `d` (today)."""
    r = random.random()
    if r < 0.18:
        fam = random.choice(FAMILY)
        amt = random.choice([500, 1000, 1500, 2000, 2500, 3000, 5000, 10000, 15000, 20000])
        aw = random.choice(VOICE_AMOUNT_WORDS[amt])
        day = random.choice(list(DAY_WORDS))
        if fam[1] in ("mummy", "papa") and random.random() < 0.5:
            text = random.choice([
                f"{fam[0]} ne {day} {aw} diye the", f"{fam[0]} se {aw} rupaye mile", f"{day} {fam[0]} ne {aw} bheje",
            ])
            lab = _label("credit", float(amt), fam[0], "cash" if "diye" in text else "upi", None, d + timedelta(days=DAY_WORDS[day]), "family_transfer")
        else:
            rel = fam[1] if fam[0] in ("Rahul", "Pooja") else ""
            who = f"{fam[0]} {rel}".strip()
            text = random.choice([
                f"{day} {who} ko {aw} bheje", f"{who} ko {aw} rupaye diye {day}", f"{who} ko {aw} transfer kiye",
            ])
            cash = "diye" in text
            lab = _label("debit", float(amt), who, "cash" if cash else "upi", None, d + timedelta(days=DAY_WORDS.get(day, 0)) if day in text else None, "family_transfer")
        return Example(text, lab, "voice_family")
    if r < 0.26:
        amt = random.choice([500, 1000, 2000, 3000, 5000, 10000])
        aw = random.choice(VOICE_AMOUNT_WORDS[amt])
        day = random.choice(list(DAY_WORDS))
        text = random.choice([f"{day} ATM se {aw} nikale", f"ATM se {aw} cash nikala {day}", f"{aw} rupaye bank se nikale"])
        dd = d + timedelta(days=DAY_WORDS[day]) if day in text else None
        return Example(text, _label("debit", float(amt), "ATM", "atm", None, dd, "cash_withdrawal"), "voice_atm")
    cat = random.choice([c for c in VOICE_MERCHANT_WORDS if c != "cash_withdrawal"])
    word = random.choice(VOICE_MERCHANT_WORDS[cat])
    amts = [a for a in VOICE_AMOUNT_WORDS if AMOUNT_RANGES[cat][0] <= a <= AMOUNT_RANGES[cat][1]] or [100]
    amt = random.choice(amts)
    aw = random.choice(VOICE_AMOUNT_WORDS[amt])
    day = random.choice(list(DAY_WORDS) + ["", ""])
    cash_phr = random.random() < 0.5
    verb = random.choice(["diye", "de diye", "cash diye"]) if cash_phr else random.choice(["bhare", "pay kiye", "UPI se diye", "online kiye", "kar diye"])
    text = random.choice([
        f"{day} {word} ko {aw} {verb}", f"{word} ke {aw} lage {day}", f"{day} {word} pe {aw} kharch hue",
        f"{word} ka {aw} {verb} {day}", f"{aw} rupaye {word} {verb}",
    ]).replace("  ", " ").strip()
    ch = "cash" if ("cash" in text or verb in ("diye", "de diye")) else ("upi" if "UPI" in text or "online" in text else "unknown")
    dd = d + timedelta(days=DAY_WORDS[day]) if (day and day in text) else None
    hints = None
    if with_hint and random.random() < 0.6:
        # user corrected this merchant before -> hint overrides the default category
        new_cat = random.choice([c for c in AMOUNT_RANGES if c not in (cat, "salary_income", "cash_withdrawal")])
        hints = [f"Treat '{word}' as {new_cat}."]
        cat = new_cat
    return Example(text, _label("debit", float(amt), word, ch, None, dd, cat), "voice_merchant", hints)


# --------------------------------------------------------------------------- noise
def noisify(text: str) -> str:
    """Cheap realistic corruption: SMS truncation, case, dropped punctuation, spaces."""
    ops = random.sample(["lower", "trunc", "nopunct", "squash", "none", "none"], k=1)
    for op in ops:
        if op == "lower":
            text = text.lower()
        elif op == "trunc" and len(text) > 90:
            cut = random.randint(int(len(text) * 0.7), len(text) - 1)
            text = text[:cut]
        elif op == "nopunct":
            text = text.replace(".", " ").replace(",", "").replace(";", " ")
        elif op == "squash":
            text = " ".join(text.split())
    return text


def _label(direction, amount, counterparty, channel, l4, d, category):
    return {
        "direction": direction,
        "amount": float(amount),
        "currency": "INR",
        "counterparty": counterparty,
        "channel": channel,
        "account_last4": l4,
        "date": d.isoformat() if d else None,
        "category": category,
    }


# --------------------------------------------------------------------------- driver
def make_example(today: date) -> Example:
    d = today - timedelta(days=random.randint(0, 120))
    l4 = last4()
    r = random.random()
    with_hint = random.random() < 0.15
    if r < 0.42:
        return sms_merchant_debit(d, random.choice([m for m in MERCHANTS if m[2] not in ("rent_emi",)]), l4, with_hint)
    if r < 0.56:
        return sms_app_notification(d, random.choice([m for m in MERCHANTS if m[2] not in ("rent_emi",)]), l4)
    if r < 0.62:
        return sms_salary(d, l4)
    if r < 0.67:
        return sms_atm(d, l4)
    if r < 0.71:
        return sms_emi(d, l4)
    if r < 0.74:
        return sms_rent_transfer(d, l4)
    if r < 0.82:
        return sms_p2p(d, l4, with_hint)
    return voice_note(today, with_hint)


def to_record(ex: Example) -> dict:
    errs = validate(ex.label)
    assert not errs, (errs, ex)
    return {
        "messages": [
            {"role": "system", "content": build_system_prompt(ex.hints)},
            {"role": "user", "content": ex.text},
            {"role": "assistant", "content": to_json(ex.label)},
        ],
        "label": ex.label,
        "kind": ex.kind,
    }


def amount_appears(ex: Example) -> bool:
    """Sanity: the labelled amount must be recoverable from the text (digits or Hinglish words)."""
    amt = ex.label["amount"]
    cands = {f"{amt:.2f}", rupees(amt, "plain"), indian_group(amt, 2)}
    if amt == int(amt):
        cands |= {indian_group(amt, 0), str(int(amt))}
    if int(amt) in VOICE_AMOUNT_WORDS:
        cands |= set(VOICE_AMOUNT_WORDS[int(amt)])
    return any(c in ex.text for c in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise", type=float, default=0.25, help="fraction of train examples to corrupt")
    ap.add_argument("--augment", action="store_true", help="add Tinker-sampled paraphrases to train (needs TINKER_API_KEY)")
    args = ap.parse_args()

    random.seed(args.seed)
    today = date(2026, 10, 2)
    total = args.train + args.val + args.test
    seen, examples = set(), []
    while len(examples) < total:
        ex = make_example(today)
        if ex.text in seen:
            continue
        if not amount_appears(ex):
            raise SystemExit(f"amount not recoverable: {ex}")
        seen.add(ex.text)
        examples.append(ex)

    random.shuffle(examples)
    test, val, train = examples[: args.test], examples[args.test : args.test + args.val], examples[args.test + args.val :]
    for ex in train:
        if random.random() < args.noise:
            ex.text = noisify(ex.text)

    if args.augment:
        from data.augment import augment_with_tinker  # lazy: needs tinker
        train += augment_with_tinker(train, n=min(400, len(train) // 3))

    for name, split in (("train", train), ("val", val), ("test", test)):
        path = HERE / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(to_record(ex), ensure_ascii=False) + "\n")
        kinds = {}
        cats = {}
        for ex in split:
            kinds[ex.kind] = kinds.get(ex.kind, 0) + 1
            cats[ex.label["category"]] = cats.get(ex.label["category"], 0) + 1
        print(f"{name}: {len(split)} -> {path}")
        print("  categories:", dict(sorted(cats.items())))
        print("  templates :", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
