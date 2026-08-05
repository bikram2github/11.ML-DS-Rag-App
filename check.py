import psycopg

conn = psycopg.connect(
    host="my-rag-app.cjequ4ik071u.eu-north-1.rds.amazonaws.com",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="Bikram12",
    sslmode="require",
    connect_timeout=10,
)

print("Connected!")
conn.close()