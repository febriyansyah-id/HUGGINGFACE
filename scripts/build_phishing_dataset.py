"""
Generator dataset sintetis dwibahasa (Indonesia 🇮🇩 + English 🇺🇸) untuk
deteksi PHISHING EMAIL.

Motivasi: corpus phishing publik yang ber-label sering privat/licin. Generator
ini memproduksi contoh-realistis berbasis pola linguistik & struktural phishing
nyata (urgensi, permintaan kredensial, spoofing institusi, URL mencurigakan,
ancaman suspensi, dst.) versus email legitimasi rutin.

Digunakan untuk:
  1. Mendemonstrasikan fine-tune pengklasifikasi biner (phiśhing ⇄ normal).
  2. Dipublikasikan sebagai dataset referensi di Hugging Face Hub.

CATATAN ETIK: Konten di-generate SEMATA untuk riset/edu defensive. Tidak
digunakan untuk kampanye nyata.
"""

from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

SEED = 2026
OUT_DIR = Path("./data/phishing_email")
CSV_PATH = OUT_DIR / "phishing_email_multilingual.csv"

INSTITUTIONS = {
    "bank": ["BCA", "Mandiri", "BNI", "Bank Central Asia", "PayPal", "Citibank"],
    "service": ["Netflix", "Apple Support", "Microsoft Account Team", "Google Cloud", "Shopee PayLater"],
}

URGENCY_ID = [
    "Akun Anda AKAN DINONAKTIFKAN dalam 24 jam jika tidak segera ditindaklanjuti.",
    "Transaksi mencurigakan terdeteksi. Segera VERIFIKASI untuk menjaga keamanan.",
    "Batasi akses Anda SEKARANG JUGA atau saldo akan dibekukan.",
    "Tanggapan dibutuhkan SELAMBATNYA hari ini untuk mencegah pemblokiran permanent.",
    "INI PEMBARUAN TERAKHIR. Kegagalan respons menyebabkan penutupan rekening.",
]

URGENCY_EN = [
    "Your account will be SUSPENDED within 24 hours unless acted upon immediately.",
    "We detected suspicious activity. Verify NOW to protect your funds.",
    "Limit your access AT ONCE or your balance will be frozen permanently.",
    "Response REQUIRED TODAY to prevent irreversible closure.",
    "THIS IS YOUR LAST REMINDER. Failure leads to termination of services.",
]

ACTION_REQUEST_ID = [
    "Mohon KLIK TAUTAN BERIKUT dan masukkan sandi serta PIN ATM Anda untuk verifikasi.",
    "Kirimkan nomor kartu, tanggal kedaluwarsa, dan CVV melalui formulir terlampir.",
    "Salin tautan ke peramban dan MASUKKAN KREDENSIAL BANK anda.",
    "Ambil langkah verifikasi dengan mengisi detail login pada portal mitra kami.",
    "Konfirmasi identitas dengan memasukkan OTP yang kami kirimkan.",
]

ACTION_REQUEST_EN = [
    "Kindly CLICK THE LINK BELOW and submit your password and debit-card PIN to verify.",
    "Reply with your card number, expiry date, and CVV via the enclosed form.",
    "Copy the link into your browser and SIGN INTO YOUR ACCOUNT CREDENTIALS.",
    "Take the verification step by entering login details on our partner portal.",
    "Confirm identity by submitting the OTP we dispatched to you.",
]

ATTACHMENT_ID = [
    "Invoice overdue terlampir (PDF). Silakan buka untuk rincian pembayaran.",
    "Bukti transfer terlampir. Mohon UNDUH dan periksa.",
    "Tagihan bulanan terlampir dalam ZIP. Ekstrak untuk melihat faktur.",
]

ATTACHMENT_EN = [
    "Attached invoice is OVERDUE (PDF). Kindly open for payment breakdown.",
    "Payment receipt attached. PLEASE DOWNLOAD and review promptly.",
    "Monthly bill zipped herein. Extract archive to view your statement.",
]

GREETINGS_OK_ID = [
    "Dear Nasabah PT Bank Mitra Utama,",
    "Halo Valued Customer,",
    "Kepada Anggota Premium Kami,",
    "Customer Service Department,",
]

GREETINGS_OK_EN = [
    "Hello Esteemed Client,",
    "Greetings Preferred Partner,",
    "Hi Loyal Subscriber,",
    "Support Resolution Center,",
]

GOODY_CONTEXT_ID = [
    "Informasi promo cashback minggu ini tersedia di situs resmi kami www.bankmitrautama.co.id.",
    "Mutasi rekening periode Januari-Maret dapat diunduh melalui aplikasi mobile resmi.",
    "Agenda webinar investasi edisi April telah dijadwalkan; RSVP via tautan kantor pusat.",
    "Surat keterangan domisili nasabah telah diperbarui di cabang terdekat.",
]

GOODY_CONTEXT_EN = [
    "Weekly cashback promotion details reside on our official site www.acmegroup.com.",
    "Statement for January-March is downloadable from our authorised mobile app.",
    "April investment seminar agenda scheduled; please RSVP via headquarters channel.",
    "Updated certificate of residence issued at nearest regional office.",
]

BADGE_GENERIC = [
    "*This mailbox is unattended. Replies bounce.*",
    "© Copyright Reserved. Confidential communication.",
    "Automatically generated notification — do not reply.",
]


def _fake_leaky_url(institution: str) -> str:
    slug = institution.lower().replace(" ", "")
    domains = [
        f"http://{slug}-secure.updater-service.online/login",
        f"http://accounts-{slug}.verify-center.cc/confirm?id=9xk2",
        f"https://support-{slug}.paymentgate-way.net/redirect",
        f"http://{slug}-portal.security-check.site/credential",
        f"https://verify-{slug}.fast-response.link/?tk=eyJhbGciOiJI",
    ]
    return random.choice(domains)


def _real_official_url(institution: str) -> str:
    slugs = {
        "BCA": "klikbca.com", "Mandiri": "bankmandiri.co.id", "BNI": "bni.co.id",
        "PayPal": "paypal.com", "Netflix": "netflix.com", "Apple Support": "apple.com",
        "Microsoft Account Team": "microsoft.com", "Google Cloud": "cloud.google.com",
        "Shopee PayLater": "shopee.co.id", "Citibank": "citi.com",
        "Bank Central Asia": "klikbca.com",
    }
    return "https://www." + slugs.get(institution, "official-domain.example")


def gen_phishing(id_: int, lang: str) -> tuple[int, str, int]:
    inst = random.choice(sum(INSTITUTIONS.values(), []))
    greeting = random.choice(GREETINGS_OK_ID if lang == "id" else GREETINGS_OK_EN)
    urgency = random.choice(URGENCY_ID if lang == "id" else URGENCY_EN)
    action = random.choice(ACTION_REQUEST_ID if lang == "id" else ACTION_REQUEST_EN)
    leaky = _fake_leaky_url(inst)
    attachment = random.choice(ATTACHMENT_ID if lang == "id" else ATTACHMENT_EN)
    closing = random.choice([
        "Thank you for cooperating. Regards, Fraud Prevention Unit.",
        "Terima kasih atas kerja samanya. Hormat kami, Divisi Anti Penipuan.",
        "Stay protected. Warm regards, Trust & Safety Office.",
    ])
    body = f"{greeting}\n\n{urgency}\n\n{action}\n\nReferensi: {attachment}\n\nTAUTAN AMAN?? --> {leaky}\n\n{closing}"
    return id_, body, 1


def gen_normal(id_: int, lang: str) -> tuple[int, str, int]:
    inst = random.choice(["PT Bank Mitra Utama", "Acme Group Corp", "Regional Utility Co"])
    ctx = random.choice(GOODY_CONTEXT_ID if lang == "id" else GOODY_CONTEXT_EN)
    footer = random.choice(BADGE_GENERIC)
    subject_area = random.choice([
        "Newsletter Bulanan Investasi",
        "Quarterly Statement Summary",
        "Maintenance Notice — Scheduled Downtime",
        "Tax Document Availability Alert",
    ])
    body = f"Subject: {subject_area}\n\n{ctx}\n\nOfficial contact: {_real_official_url(inst)}\n\n{footer}"
    return id_, body, 0


def build_csv(target_rows: int = 600, ratio_phish: float = 0.5) -> None:
    rnd = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, str, str, int]] = []
    counter = itertools.count(start=1)

    n_phish = int(target_rows * ratio_phish)
    n_norm = target_rows - n_phish

    for _ in range(n_phish):
        rid = next(counter)
        lang = rnd.choice(["id", "en"])
        pid, body, lbl = gen_phishing(rid, lang)
        rows.append((pid, lang, body, lbl))

    for _ in range(n_norm):
        rid = next(counter)
        lang = rnd.choice(["id", "en"])
        pid, body, lbl = gen_normal(rid, lang)
        rows.append((pid, lang, body, lbl))

    rnd.shuffle(rows)

    header = ["id", "language", "text", "label"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    pos = sum(1 for *_ , lb in rows if lb == 1)
    langs = {lg for (_ , lg, _b, _lb) in rows}
    print(f"Written {len(rows)} rows -> {CSV_PATH}")
    print(f"  phishing:{pos}  normal:{len(rows)-pos}  langs:{sorted(langs)}")


if __name__ == "__main__":
    build_csv()
