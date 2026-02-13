"""
Convert external bot detection datasets to the competition format.

Competition format:
{
  "id": dataset_id,
  "lang": "en" or "fr",
  "metadata": {
    "start_time": "...",
    "end_time": "...",
    "total_amount_users": N,
    "total_amount_posts": N,
    "topics": [...],
    "users_average_amount_posts": float,
    "users_average_z_score": float
  },
  "posts": [...],
  "users": [...]
}

Bot list format:
- One user ID per line
"""

import json
import os
import uuid
import random
import zipfile
import csv
import math
from datetime import datetime, timedelta


def generate_uuid():
    """Generate a UUID string."""
    return str(uuid.uuid4())


def calculate_z_score(value, mean, std):
    """Calculate z-score for a value."""
    if std == 0 or std is None:
        return 0.0
    return (value - mean) / std


def read_csv_from_zip(zip_path):
    """Read CSV file from zip archive."""
    if not os.path.exists(zip_path):
        return []
    
    rows = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return []
            
            with z.open(csv_files[0]) as f:
                # Read as text
                content = f.read().decode('utf-8', errors='ignore')
                lines = content.split('\n')
                if not lines:
                    return []
                
                # Parse CSV
                reader = csv.DictReader(lines)
                for row in reader:
                    rows.append(row)
    except Exception as e:
        print(f"    Error reading {zip_path}: {e}")
    
    return rows


def convert_cresci_2015(input_dir):
    """Convert Cresci-2015 dataset."""
    print("Converting Cresci-2015 dataset...")
    
    files = {
        'E13': 'genuine',
        'TFP': 'genuine',
        'FSF': 'bot',
        'INT': 'bot',
        'TWT': 'bot'
    }
    
    user_profiles = {}
    bot_ids = []
    
    for prefix, label in files.items():
        zip_path = os.path.join(input_dir, f'{prefix}.csv.zip')
        rows = read_csv_from_zip(zip_path)
        
        if not rows:
            print(f"  Warning: No data for {prefix}")
            continue
        
        print(f"  Processing {prefix}: {len(rows)} users")
        
        for row in rows:
            user_id = str(row.get('id', row.get('user_id', generate_uuid())))
            if user_id not in user_profiles:
                try:
                    tweet_count = int(row.get('statuses_count', row.get('tweet_count', random.randint(10, 100))))
                except:
                    tweet_count = random.randint(10, 100)
                
                user_profiles[user_id] = {
                    'id': user_id,
                    'label': label,
                    'username': str(row.get('screen_name', row.get('username', f'user_{user_id}'))),
                    'name': str(row.get('name', '')),
                    'description': str(row.get('description', '')),
                    'location': str(row.get('location', '')),
                    'tweet_count': tweet_count,
                    'created_at': str(row.get('created_at', ''))
                }
                if label == 'bot':
                    bot_ids.append(user_id)
    
    if not user_profiles:
        print("  No data found for Cresci-2015")
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=15,
        lang='en',
        description='Cresci-2015: Genuine accounts and fake followers'
    )


def convert_cresci_2017(input_dir):
    """Convert Cresci-2017 dataset."""
    print("Converting Cresci-2017 dataset...")
    
    files = {
        'genuine_accounts.csv': 'genuine',
        'social_spambots_1.csv': 'bot',
        'social_spambots_2.csv': 'bot',
        'social_spambots_3.csv': 'bot',
        'traditional_spambots_1.csv': 'bot',
        'traditional_spambots_2.csv': 'bot',
        'traditional_spambots_3.csv': 'bot',
        'traditional_spambots_4.csv': 'bot',
        'fake_followers.csv': 'bot'
    }
    
    user_profiles = {}
    bot_ids = []
    
    for filename, label in files.items():
        zip_path = os.path.join(input_dir, f'{filename}.zip')
        rows = read_csv_from_zip(zip_path)
        
        if not rows:
            continue
        
        print(f"  Processing {filename}: {len(rows)} users")
        
        for row in rows:
            user_id = str(row.get('id', row.get('user_id', generate_uuid())))
            if user_id not in user_profiles:
                try:
                    tweet_count = int(row.get('statuses_count', row.get('tweet_count', random.randint(10, 100))))
                except:
                    tweet_count = random.randint(10, 100)
                
                user_profiles[user_id] = {
                    'id': user_id,
                    'label': label,
                    'username': str(row.get('screen_name', row.get('username', f'user_{user_id}'))),
                    'name': str(row.get('name', '')),
                    'description': str(row.get('description', '')),
                    'location': str(row.get('location', '')),
                    'tweet_count': tweet_count,
                    'created_at': str(row.get('created_at', ''))
                }
                if label == 'bot':
                    bot_ids.append(user_id)
    
    if not user_profiles:
        print("  No data found for Cresci-2017")
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=17,
        lang='en',
        description='Cresci-2017: Genuine, traditional spambots, and social spambots'
    )


def convert_varol_2017(input_file):
    """Convert Varol-2017 dataset."""
    print("Converting Varol-2017 dataset...")
    
    if not os.path.exists(input_file):
        print(f"  File not found: {input_file}")
        return None
    
    user_profiles = {}
    bot_ids = []
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = parts[0]
                label = 'bot' if parts[1] == '1' else 'genuine'
                
                user_profiles[user_id] = {
                    'id': user_id,
                    'label': label,
                    'username': f'user_{user_id}',
                    'name': '',
                    'description': '',
                    'location': '',
                    'tweet_count': random.randint(10, 100),
                    'created_at': ''
                }
                if label == 'bot':
                    bot_ids.append(user_id)
    
    print(f"  Processed {len(user_profiles)} users")
    
    if not user_profiles:
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=1701,
        lang='en',
        description='Varol-2017: Manually annotated human and bot accounts'
    )


def convert_gilani_2017(tsv_file):
    """Convert Gilani-2017 dataset."""
    print("Converting Gilani-2017 dataset...")
    
    if not os.path.exists(tsv_file):
        print(f"  TSV file not found: {tsv_file}")
        return None
    
    user_profiles = {}
    bot_ids = []
    
    with open(tsv_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                user_id = parts[0]
                label = 'bot' if parts[1] == 'bot' else 'genuine'
                
                user_profiles[user_id] = {
                    'id': user_id,
                    'label': label,
                    'username': f'user_{user_id}',
                    'name': '',
                    'description': '',
                    'location': '',
                    'tweet_count': random.randint(10, 100),
                    'created_at': ''
                }
                if label == 'bot':
                    bot_ids.append(user_id)
    
    print(f"  Processed {len(user_profiles)} users")
    
    if not user_profiles:
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=1702,
        lang='en',
        description='Gilani-2017: Manually annotated human and bot accounts'
    )


def convert_caverlee_2011(input_dir):
    """Convert Caverlee-2011 social honeypot dataset."""
    print("Converting Caverlee-2011 dataset...")
    
    honeypot_dir = os.path.join(input_dir, 'social_honeypot_icwsm_2011')
    if not os.path.exists(honeypot_dir):
        print(f"  Directory not found: {honeypot_dir}")
        return None
    
    user_profiles = {}
    bot_ids = []
    
    # Read content polluters (bots)
    polluters_file = os.path.join(honeypot_dir, 'content_polluters.txt')
    if os.path.exists(polluters_file):
        with open(polluters_file, 'r', encoding='latin1', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    user_id = parts[0]
                    try:
                        tweet_count = int(parts[4]) if len(parts) > 4 else random.randint(10, 100)
                    except:
                        tweet_count = random.randint(10, 100)
                    
                    user_profiles[user_id] = {
                        'id': user_id,
                        'label': 'bot',
                        'username': f'user_{user_id}',
                        'name': '',
                        'description': '',
                        'location': '',
                        'tweet_count': tweet_count,
                        'created_at': ''
                    }
                    bot_ids.append(user_id)
    
    # Read legitimate users
    legitimate_file = os.path.join(honeypot_dir, 'legitimate_users.txt')
    if os.path.exists(legitimate_file):
        with open(legitimate_file, 'r', encoding='latin1', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    user_id = parts[0]
                    if user_id not in user_profiles:
                        try:
                            tweet_count = int(parts[4]) if len(parts) > 4 else random.randint(10, 100)
                        except:
                            tweet_count = random.randint(10, 100)
                        
                        user_profiles[user_id] = {
                            'id': user_id,
                            'label': 'genuine',
                            'username': f'user_{user_id}',
                            'name': '',
                            'description': '',
                            'location': '',
                            'tweet_count': tweet_count,
                            'created_at': ''
                        }
    
    print(f"  Processed {len(user_profiles)} users")
    
    if not user_profiles:
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=1101,
        lang='en',
        description='Caverlee-2011: Social honeypot dataset with content polluters and legitimate users'
    )


def convert_midterm_2018(tsv_file):
    """Convert Midterm-2018 dataset."""
    print("Converting Midterm-2018 dataset...")
    
    if not os.path.exists(tsv_file):
        print(f"  TSV file not found: {tsv_file}")
        return None
    
    user_profiles = {}
    bot_ids = []
    
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            user_id = row.get('user_id', generate_uuid())
            label = 'bot' if row.get('label', '').lower() in ['bot', '1'] else 'genuine'
            
            try:
                tweet_count = int(row.get('statuses_count', random.randint(10, 100)))
            except:
                tweet_count = random.randint(10, 100)
            
            user_profiles[user_id] = {
                'id': user_id,
                'label': label,
                'username': row.get('screen_name', f'user_{user_id}'),
                'name': row.get('name', ''),
                'description': row.get('description', ''),
                'location': row.get('location', ''),
                'tweet_count': tweet_count,
                'created_at': row.get('created_at', '')
            }
            if label == 'bot':
                bot_ids.append(user_id)
    
    print(f"  Processed {len(user_profiles)} users")
    
    if not user_profiles:
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=1801,
        lang='en',
        description='Midterm-2018: Manually labeled accounts from 2018 US midterm elections'
    )


def convert_verified_2019(tsv_file):
    """Convert Verified-2019 dataset (all humans)."""
    print("Converting Verified-2019 dataset...")
    
    if not os.path.exists(tsv_file):
        print(f"  TSV file not found: {tsv_file}")
        return None
    
    user_profiles = {}
    bot_ids = []  # All verified = humans
    
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            user_id = row.get('user_id', generate_uuid())
            
            try:
                tweet_count = int(row.get('statuses_count', random.randint(10, 100)))
            except:
                tweet_count = random.randint(10, 100)
            
            user_profiles[user_id] = {
                'id': user_id,
                'label': 'genuine',
                'username': row.get('screen_name', f'user_{user_id}'),
                'name': row.get('name', ''),
                'description': row.get('description', ''),
                'location': row.get('location', ''),
                'tweet_count': tweet_count,
                'created_at': row.get('created_at', '')
            }
    
    print(f"  Processed {len(user_profiles)} users (all genuine)")
    
    if not user_profiles:
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=1901,
        lang='en',
        description='Verified-2019: Verified human accounts'
    )


def convert_botwiki_2019(tsv_file):
    """Convert Botwiki-2019 dataset (all bots)."""
    print("Converting Botwiki-2019 dataset...")
    
    if not os.path.exists(tsv_file):
        print(f"  TSV file not found: {tsv_file}")
        return None
    
    user_profiles = {}
    bot_ids = []
    
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            user_id = row.get('user_id', generate_uuid())
            
            try:
                tweet_count = int(row.get('statuses_count', random.randint(10, 100)))
            except:
                tweet_count = random.randint(10, 100)
            
            user_profiles[user_id] = {
                'id': user_id,
                'label': 'bot',
                'username': row.get('screen_name', f'user_{user_id}'),
                'name': row.get('name', ''),
                'description': row.get('description', ''),
                'location': row.get('location', ''),
                'tweet_count': tweet_count,
                'created_at': row.get('created_at', '')
            }
            bot_ids.append(user_id)
    
    print(f"  Processed {len(user_profiles)} users (all bots)")
    
    if not user_profiles:
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=1902,
        lang='en',
        description='Botwiki-2019: Self-identified bots from botwiki.org'
    )


def convert_pronbots_2019(tsv_file):
    """Convert Pronbots-2019 dataset (all bots)."""
    print("Converting Pronbots-2019 dataset...")
    
    if not os.path.exists(tsv_file):
        print(f"  TSV file not found: {tsv_file}")
        return None
    
    user_profiles = {}
    bot_ids = []
    
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            user_id = row.get('user_id', generate_uuid())
            
            try:
                tweet_count = int(row.get('statuses_count', random.randint(10, 100)))
            except:
                tweet_count = random.randint(10, 100)
            
            user_profiles[user_id] = {
                'id': user_id,
                'label': 'bot',
                'username': row.get('screen_name', f'user_{user_id}'),
                'name': row.get('name', ''),
                'description': row.get('description', ''),
                'location': row.get('location', ''),
                'tweet_count': tweet_count,
                'created_at': row.get('created_at', '')
            }
            bot_ids.append(user_id)
    
    print(f"  Processed {len(user_profiles)} users (all bots)")
    
    if not user_profiles:
        return None
    
    return create_dataset_output(
        user_profiles, {}, bot_ids,
        dataset_id=1903,
        lang='en',
        description='Pronbots-2019: Pronbots dataset'
    )


def create_dataset_output(user_profiles, tweets, bot_ids, dataset_id, lang, description):
    """Create the standardized dataset output format."""
    # Calculate tweet counts and z-scores
    tweet_counts = [u['tweet_count'] for u in user_profiles.values()]
    mean_tweets = sum(tweet_counts) / len(tweet_counts) if tweet_counts else 0
    std_tweets = math.sqrt(sum((x - mean_tweets) ** 2 for x in tweet_counts) / len(tweet_counts)) if tweet_counts else 1
    
    if std_tweets == 0:
        std_tweets = 1
    
    # Create users list with z-scores
    users = []
    for user_id, profile in user_profiles.items():
        z_score = calculate_z_score(profile['tweet_count'], mean_tweets, std_tweets)
        location = profile['location']
        if location == '' or location is None:
            location = None
        
        users.append({
            'id': profile['id'],
            'tweet_count': profile['tweet_count'],
            'z_score': z_score,
            'username': profile['username'],
            'name': profile['name'],
            'description': profile['description'],
            'location': location
        })
    
    # Create synthetic posts
    posts = []
    base_time = datetime(2020, 1, 1)
    
    for user in users:
        num_posts = min(user['tweet_count'], 100)  # Max 100 posts per user
        for i in range(num_posts):
            posts.append({
                'text': f'Sample post {i} from user {user["username"]}',
                'created_at': (base_time + timedelta(hours=random.randint(0, 1000))).isoformat() + 'Z',
                'id': generate_uuid(),
                'author_id': user['id'],
                'lang': lang
            })
    
    # Calculate metadata
    total_posts = len(posts)
    total_users = len(users)
    avg_posts = total_posts / total_users if total_users > 0 else 0
    avg_z_score = sum(u['z_score'] for u in users) / len(users) if users else 0
    
    dataset = {
        'id': dataset_id,
        'lang': lang,
        'metadata': {
            'start_time': '2020-01-01T00:00:00Z',
            'end_time': '2020-12-31T23:59:59Z',
            'total_amount_users': total_users,
            'total_amount_posts': total_posts,
            'topics': [{'topic': 'general', 'keywords': ['twitter', 'social media']}],
            'users_average_amount_posts': avg_posts,
            'users_average_z_score': avg_z_score,
            'description': description,
            'source': 'Converted from academic dataset'
        },
        'posts': posts,
        'users': users
    }
    
    return dataset, bot_ids


def save_dataset(dataset, bot_ids, output_dir, name):
    """Save dataset to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main dataset
    output_file = os.path.join(output_dir, f'dataset.posts&users.{name}.json')
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Save bot list
    bot_file = os.path.join(output_dir, f'dataset.bots.{name}.txt')
    with open(bot_file, 'w') as f:
        for bot_id in bot_ids:
            f.write(f'{bot_id}\n')
    
    print(f"  Saved to {output_file}")
    print(f"  Bots: {len(bot_ids)} / {len(dataset['users'])} users")


def main():
    """Main conversion function."""
    base_input = '/Users/q/Desktop/projects/bot-or-not/dataset_scaped'
    base_output = '/Users/q/Desktop/projects/bot-or-not/dataset_external'
    
    os.makedirs(base_output, exist_ok=True)
    
    converted = []
    failed = []
    
    # 1. Cresci-2015
    try:
        result = convert_cresci_2015(os.path.join(base_input, 'datasets_full.csv'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'cresci15')
            converted.append('Cresci-2015')
        else:
            failed.append('Cresci-2015')
    except Exception as e:
        print(f"  Error converting Cresci-2015: {e}")
        failed.append('Cresci-2015')
    
    print()
    
    # 2. Cresci-2017
    try:
        result = convert_cresci_2017(os.path.join(base_input, 'datasets_full.csv'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'cresci17')
            converted.append('Cresci-2017')
        else:
            failed.append('Cresci-2017')
    except Exception as e:
        print(f"  Error converting Cresci-2017: {e}")
        failed.append('Cresci-2017')
    
    print()
    
    # 3. Varol-2017
    try:
        result = convert_varol_2017(os.path.join(base_input, 'varol-2017.dat'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'varol17')
            converted.append('Varol-2017')
        else:
            failed.append('Varol-2017')
    except Exception as e:
        print(f"  Error converting Varol-2017: {e}")
        failed.append('Varol-2017')
    
    print()
    
    # 4. Gilani-2017
    try:
        result = convert_gilani_2017(os.path.join(base_input, 'gilani-2017.tsv'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'gilani17')
            converted.append('Gilani-2017')
        else:
            failed.append('Gilani-2017')
    except Exception as e:
        print(f"  Error converting Gilani-2017: {e}")
        failed.append('Gilani-2017')
    
    print()
    
    # 5. Caverlee-2011
    try:
        result = convert_caverlee_2011(base_input)
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'caverlee11')
            converted.append('Caverlee-2011')
        else:
            failed.append('Caverlee-2011')
    except Exception as e:
        print(f"  Error converting Caverlee-2011: {e}")
        failed.append('Caverlee-2011')
    
    print()
    
    # 6. Midterm-2018
    try:
        result = convert_midterm_2018(os.path.join(base_input, 'midterm-2018.tsv'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'midterm18')
            converted.append('Midterm-2018')
        else:
            failed.append('Midterm-2018')
    except Exception as e:
        print(f"  Error converting Midterm-2018: {e}")
        failed.append('Midterm-2018')
    
    print()
    
    # 7. Verified-2019 (all humans)
    try:
        result = convert_verified_2019(os.path.join(base_input, 'verified-2019.tsv'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'verified19')
            converted.append('Verified-2019')
        else:
            failed.append('Verified-2019')
    except Exception as e:
        print(f"  Error converting Verified-2019: {e}")
        failed.append('Verified-2019')
    
    print()
    
    # 8. Botwiki-2019 (all bots)
    try:
        result = convert_botwiki_2019(os.path.join(base_input, 'botwiki-2019.tsv'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'botwiki19')
            converted.append('Botwiki-2019')
        else:
            failed.append('Botwiki-2019')
    except Exception as e:
        print(f"  Error converting Botwiki-2019: {e}")
        failed.append('Botwiki-2019')
    
    print()
    
    # 9. Pronbots-2019 (all bots)
    try:
        result = convert_pronbots_2019(os.path.join(base_input, 'pronbots-2019.tsv'))
        if result:
            dataset, bot_ids = result
            save_dataset(dataset, bot_ids, base_output, 'pronbots19')
            converted.append('Pronbots-2019')
        else:
            failed.append('Pronbots-2019')
    except Exception as e:
        print(f"  Error converting Pronbots-2019: {e}")
        failed.append('Pronbots-2019')
    
    # Summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Successfully converted: {len(converted)}")
    for name in converted:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed/Skipped: {len(failed)}")
        for name in failed:
            print(f"  ✗ {name}")


if __name__ == '__main__':
    main()
