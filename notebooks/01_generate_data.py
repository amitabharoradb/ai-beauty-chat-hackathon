# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install faker==24.0.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
import json
import random
from datetime import datetime, timedelta
from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
fake = Faker()
Faker.seed(42)
random.seed(42)

UC_CATALOG = os.getenv("UC_CATALOG", "amitabh_arora_catalog")  # catalog already exists
UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora_hackathon")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")

# COMMAND ----------
# Categories
CATEGORIES = [
    {"id": "cat_skincare", "name": "Skincare"},
    {"id": "cat_makeup",   "name": "Makeup"},
    {"id": "cat_haircare", "name": "Haircare"},
]

# Products — 5-15 per category (Sephora-inspired)
PRODUCTS = [
    # Skincare (7)
    {"id":"prod_sk_001","category_id":"cat_skincare","name":"Hydra-Gel Cleanser","description":"Lightweight gel cleanser for oily and combination skin","price":38.0,"key_ingredients":["niacinamide","salicylic acid"],"benefits":["deep cleanse","pore minimizing"],"tags":["oily skin","acne-prone","vegan"]},
    {"id":"prod_sk_002","category_id":"cat_skincare","name":"Glow Vitamin C Serum","description":"Brightening serum with 15% vitamin C","price":54.0,"key_ingredients":["vitamin C","ferulic acid","hyaluronic acid"],"benefits":["brightening","anti-aging","hydration"],"tags":["all skin","vegan","fragrance-free"]},
    {"id":"prod_sk_003","category_id":"cat_skincare","name":"Barrier Repair Moisturizer","description":"Rich cream that restores the skin barrier","price":46.0,"key_ingredients":["ceramides","squalane","peptides"],"benefits":["hydration","barrier repair","calming"],"tags":["dry skin","sensitive skin","fragrance-free"]},
    {"id":"prod_sk_004","category_id":"cat_skincare","name":"Daily Defense SPF 50","description":"Lightweight sunscreen with no white cast","price":34.0,"key_ingredients":["zinc oxide","niacinamide"],"benefits":["UV protection","pore minimizing"],"tags":["all skin","vegan","water-resistant"]},
    {"id":"prod_sk_005","category_id":"cat_skincare","name":"Pore Refining Toner","description":"BHA toner that unclogs pores and balances skin","price":28.0,"key_ingredients":["2% BHA","witch hazel","aloe vera"],"benefits":["pore minimizing","exfoliating","balancing"],"tags":["oily skin","acne-prone","vegan"]},
    {"id":"prod_sk_006","category_id":"cat_skincare","name":"Firming Eye Cream","description":"Peptide-rich cream for dark circles and puffiness","price":62.0,"key_ingredients":["retinol","caffeine","peptides"],"benefits":["anti-aging","brightening","depuffing"],"tags":["all skin","fragrance-free"]},
    {"id":"prod_sk_007","category_id":"cat_skincare","name":"Overnight Recovery Mask","description":"Sleeping mask that repairs skin overnight","price":48.0,"key_ingredients":["bakuchiol","niacinamide","ceramides"],"benefits":["repair","hydration","glow"],"tags":["all skin","vegan","fragrance-free"]},

    # Makeup (7)
    {"id":"prod_mk_001","category_id":"cat_makeup","name":"Skin Tint Foundation SPF 20","description":"Lightweight buildable coverage with SPF","price":44.0,"key_ingredients":["hyaluronic acid","zinc oxide"],"benefits":["natural finish","SPF","hydrating"],"tags":["all skin","vegan","buildable"]},
    {"id":"prod_mk_002","category_id":"cat_makeup","name":"Precision Concealer","description":"Full coverage concealer for dark circles and blemishes","price":32.0,"key_ingredients":["vitamin E","niacinamide"],"benefits":["full coverage","long-wearing","brightening"],"tags":["all skin","vegan"]},
    {"id":"prod_mk_003","category_id":"cat_makeup","name":"Lash Amplify Mascara","description":"Volumizing mascara for bold lashes","price":26.0,"key_ingredients":["argan oil","biotin"],"benefits":["volume","length","conditioning"],"tags":["all lash types","cruelty-free"]},
    {"id":"prod_mk_004","category_id":"cat_makeup","name":"Satin Lip Color","description":"Comfortable satin-finish lipstick in 12 shades","price":24.0,"key_ingredients":["jojoba oil","vitamin E"],"benefits":["hydrating","pigmented","long-wearing"],"tags":["vegan","fragrance-free"]},
    {"id":"prod_mk_005","category_id":"cat_makeup","name":"Cream Blush Stick","description":"Buildable cream blush in 8 natural shades","price":28.0,"key_ingredients":["shea butter","rose extract"],"benefits":["natural flush","blendable","hydrating"],"tags":["all skin","vegan","fragrance-free"]},
    {"id":"prod_mk_006","category_id":"cat_makeup","name":"Liquid Highlighter","description":"Radiant liquid highlighter for glass skin","price":36.0,"key_ingredients":["pearl extract","hyaluronic acid"],"benefits":["radiance","hydration","buildable glow"],"tags":["all skin","vegan"]},
    {"id":"prod_mk_007","category_id":"cat_makeup","name":"Eyeshadow Palette - Nudes","description":"12-shade neutral palette for everyday looks","price":52.0,"key_ingredients":["vitamin E"],"benefits":["versatile","blendable","long-wearing"],"tags":["vegan","cruelty-free"]},

    # Haircare (6)
    {"id":"prod_hc_001","category_id":"cat_haircare","name":"Hydrating Shampoo","description":"Sulfate-free shampoo for dry and damaged hair","price":28.0,"key_ingredients":["argan oil","keratin","biotin"],"benefits":["hydration","repair","shine"],"tags":["dry hair","color-safe","vegan"]},
    {"id":"prod_hc_002","category_id":"cat_haircare","name":"Repair Conditioner","description":"Deep conditioning treatment for smooth, frizz-free hair","price":28.0,"key_ingredients":["shea butter","amino acids"],"benefits":["smoothing","hydration","frizz control"],"tags":["all hair types","vegan","sulfate-free"]},
    {"id":"prod_hc_003","category_id":"cat_haircare","name":"Intense Hair Mask","description":"Weekly treatment mask for intense repair","price":42.0,"key_ingredients":["coconut oil","keratin","biotin"],"benefits":["deep repair","shine","strengthening"],"tags":["damaged hair","color-safe","vegan"]},
    {"id":"prod_hc_004","category_id":"cat_haircare","name":"Scalp Revival Serum","description":"Balancing serum for a healthy scalp","price":38.0,"key_ingredients":["salicylic acid","peppermint","niacinamide"],"benefits":["scalp balance","soothing","growth support"],"tags":["oily scalp","vegan","fragrance-free"]},
    {"id":"prod_hc_005","category_id":"cat_haircare","name":"Curl Defining Cream","description":"Anti-frizz cream that defines and holds curls","price":32.0,"key_ingredients":["flaxseed","shea butter","aloe vera"],"benefits":["curl definition","frizz control","hydration"],"tags":["curly hair","vegan","sulfate-free"]},
    {"id":"prod_hc_006","category_id":"cat_haircare","name":"Dry Shampoo Powder","description":"Invisible dry shampoo that refreshes and volumizes","price":22.0,"key_ingredients":["rice starch","kaolin clay"],"benefits":["volume","oil absorption","refreshing"],"tags":["all hair","vegan","travel-friendly"]},
]

# COMMAND ----------
# Customers (10,000)
SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive"]
SKIN_TONES = ["fair", "light", "medium", "tan", "deep"]
CONCERNS_LIST = ["acne", "dryness", "anti-aging", "hyperpigmentation", "redness", "enlarged pores", "dark circles", "frizz"]
INTERACTION_TYPES = ["purchased", "viewed", "liked"]

customers = []
for i in range(10000):
    customer_id = f"cust_{i+1:05d}"
    customers.append({
        "id": customer_id,
        "name": fake.name(),
        "email": fake.email(),
        "age": random.randint(18, 65),
        "skin_type": random.choice(SKIN_TYPES),
        "skin_tone": random.choice(SKIN_TONES),
        "concerns": json.dumps(random.sample(CONCERNS_LIST, k=random.randint(1, 3))),
    })

# COMMAND ----------
# Customer-product interactions (~50K rows)
product_ids = [p["id"] for p in PRODUCTS]
interactions = []
for cust in customers:
    n = random.randint(2, 8)
    for prod_id in random.sample(product_ids, k=min(n, len(product_ids))):
        interactions.append({
            "customer_id": cust["id"],
            "product_id": prod_id,
            "interaction_type": random.choice(INTERACTION_TYPES),
        })

# COMMAND ----------
# Write to Unity Catalog
def write_table(data, schema, table_name, mode="overwrite"):
    df = spark.createDataFrame(data)
    df.write.format("delta").mode(mode).saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.{table_name}")
    count = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.{table_name}").count()
    print(f"  {table_name}: {count} rows")

print("Writing tables to Unity Catalog...")
write_table(CATEGORIES, None, "categories")
write_table(PRODUCTS, None, "products")
write_table(customers, None, "customers")
write_table(interactions, None, "customer_products")
print("Done.")
