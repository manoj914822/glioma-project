import sqlite3
try:
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    # Check users
    cursor.execute('SELECT id, name, email FROM users')
    users = cursor.fetchall()
    print("Users in database:")
    for user in users:
        print(f"ID: {user[0]}, Name: {user[1]}, Email: {user[2]}")

    # Check if User 3 exists
    cursor.execute('SELECT COUNT(*) FROM users WHERE id = 3')
    user3_exists = cursor.fetchone()[0]
    print(f"\nUser 3 exists: {bool(user3_exists)}")

    # Check User 3's predictions
    if user3_exists:
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = 3')
        pred_count = cursor.fetchone()[0]
        print(f"User 3 prediction count: {pred_count}")

        if pred_count > 0:
            cursor.execute('SELECT filename, predicted_class, accuracy, timestamp FROM predictions WHERE user_id = 3 ORDER BY timestamp DESC')
            predictions = cursor.fetchall()
            print("User 3 predictions:")
            for pred in predictions:
                print(f"  File: {pred[0]}, Class: {pred[1]}, Accuracy: {pred[2]}, Time: {pred[3]}")

    conn.close()
except Exception as e:
    print(f"Error: {e}")
