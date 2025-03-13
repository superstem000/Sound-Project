from flask import Flask, request, jsonify
import json
import os
import pandas as pd
import nltk
import spacy
from nltk.stem import PorterStemmer
from datetime import datetime, timedelta
from transformers import pipeline
import boto3
import os
import time
import botocore.exceptions
from boto3.dynamodb.conditions import Attr
import sys
import argparse

def read_cfg(file_path):
    config = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue            
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip().strip("'\"") 
    return config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HARDWARE_DIR = os.path.dirname(SCRIPT_DIR)

cfg_path = os.path.join(HARDWARE_DIR, "application.cfg")
config = read_cfg(cfg_path)
AWS_ACCESS_KEY_ID = config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = config.get('AWS_REGION')
CHUNKSIZE = 15 # sec

dynamodb = boto3.resource(
    'dynamodb',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

table = dynamodb.Table('respeaker_data')

# Initialize NLTK stemmer
stemmer = PorterStemmer()

app = Flask(__name__)

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Load LLM
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def word_to_num(word):
    mapping = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 
        'nine': 9, 'ten': 10, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return mapping.get(word.lower(), 0)


def get_id_str(DIR_NAME):
    ID_file  = DIR_NAME + '/assign_speaker/ID.json'
    # Load IDs from the ID file
    with open(ID_file, 'r') as f:
        ID_data = json.load(f)
        # Convert word-based numeric IDs to integers and sort them
        numeric_ids = sorted([word_to_num(info['ID'][0]) for info in ID_data.values()])
        id_str = '_'.join(map(str, numeric_ids))
    
    id_str = "Apple"
    return id_str


# Function to get the lemma (base form) of a word
def get_lemma(word):
    doc = nlp(word)
    return doc[0].lemma_

@app.route('/check_speakers_not_spoken', methods=['POST'])
def check_speakers_not_spoken():
    parser = argparse.ArgumentParser(description="directory")
    parser.add_argument("-d", "--directory", required=True, help="directory that will contain the dataset")
    args = parser.parse_args()
    DIR_NAME = args.directory
    # Get the start and end times from the request

    # Load the JSON data from the ID.json file
    with open(DIR_NAME + '/assign_speaker/ID.json', 'r') as file:
        data = json.load(file)

    # Initialize an empty set for preset_speakers
    preset_speakers = set()

    # Iterate through the data and add the first value of the ID array for each person to the set
    for person in data.values():
        if person['ID']:  # Check if the ID list is not empty
            preset_speakers.add(person['ID']) 

    data = request.json
    start_time = int(data['start_time'])  # Example format: '20'
    end_time = int(data['end_time']) #60

    
    print("Current time to call", end_time)
    # Call a function to check speakers who have not spoken within the specified time frame
    speakers_not_spoken = check_speakers_within_timeframe(start_time, end_time, preset_speakers)

    speakers_not_spoken_result = json.dumps(speakers_not_spoken)
    
    # DynamoDB update logic
    id_str = get_id_str(DIR_NAME)
    cur_date_formatted = datetime.now().strftime('%Y-%m-%d')
    cur_time_formatted = datetime.now().strftime('%H:%M:%S')

    try:
        # Fetch the current item based on group_id
        response = table.get_item(Key={'group_id': id_str})
        item = response.get('Item')

        if item:
            # Ensure current date's map exists directly within the item
            if cur_date_formatted not in item:
                item[cur_date_formatted] = {}

            # Add the new result to the 'check_speakers_not_spoken' section within the current date
            cur_date_data = item[cur_date_formatted]
            cur_date_data.setdefault('check_speakers_not_spoken', {})
            cur_date_data['check_speakers_not_spoken'][f"Results_{cur_time_formatted}"] = speakers_not_spoken_result

            # Update the item in the table
            table.put_item(Item=item)

        # Note: This else block can probably be removed since this route is called every 60 seconds, and the words_concat route is called
        # every 15 seconds, so the item will ALWAYS be there
        else:
            # If the item does not exist, create a new one with the date directly under group_id
            new_item = {
                'group_id': id_str,
                cur_date_formatted: {
                    'check_speakers_not_spoken': {
                        f"Results_{cur_time_formatted}": speakers_not_spoken_result
                    }
                }
            }
            table.put_item(Item=new_item)

    except botocore.exceptions.ClientError as error:
        # Handle the exception
        print(f"An error occurred: {error}")

    result = {
        'message': 'Check for speakers not spoken completed and stored in DynamoDB.',
        'speakers_not_spoken': speakers_not_spoken  # Assuming speakers_not_spoken is the list of speakers
    }

    print(speakers_not_spoken)
    print('test')
    print(jsonify(result))
    
    return jsonify(result)

def check_speakers_within_timeframe(start_time, end_time, preset_speakers):
    parser = argparse.ArgumentParser(description="directory")
    parser.add_argument("-d", "--directory", required=True, help="directory that will contain the dataset")
    args = parser.parse_args()
    DIR_NAME = args.directory

    speakers_not_spoken = set(preset_speakers)

    # Define the range of numbers (10 to 240) for the filenames
    for number in range(start_time, end_time + 1, CHUNKSIZE):
        # Provide path to transcript chunks here
        filename = DIR_NAME + '/recorded_data/chunk_%d.wav.json'%number
        if os.path.exists(filename):
            # Load the JSON data from the file
            with open(filename, 'r') as file:
                data = json.load(file)

                for segment in data['transcription']:
                    speaker_name = segment['speaker']
                    speakers_not_spoken.discard(speaker_name)

    print(list(speakers_not_spoken))
    return list(speakers_not_spoken)
  

@app.route('/analysis', methods=['POST'])
def analyze_transcripts():
    parser = argparse.ArgumentParser(description="directory")
    parser.add_argument("-d", "--directory", required=True, help="directory that will contain the dataset")
    args = parser.parse_args()
    DIR_NAME = args.directory

    # Load the JSON data from the ID.json file
    with open(DIR_NAME + '/assign_speaker/ID.json', 'r') as file:
        data = json.load(file)

    # Initialize an empty set for preset_speakers
    preset_speakers = set()

    # Iterate through the data and add the first value of the ID array for each person to the set
    for person in data.values():
        if person['ID']:  # Check if the ID list is not empty
            preset_speakers.add(person['ID']) 

    data = request.json
    
    total_files = int(data['total_files'])  
    x = total_files #last chunk ID + 1
    print("last_chunk_id", x)
    x += 1
    # Initialize an empty list to store the table data
    all_table_data = []
    
    # Define the range of numbers (105 to 240) for the filenames
    for number in range(CHUNKSIZE, x, CHUNKSIZE):
        #Provide path to transcript chunks here
        filename = DIR_NAME + '/recorded_data/chunk_%d.wav.json'%number
        if os.path.exists(filename):
            # Load the JSON data from the file
            with open(filename, 'r') as file:
                data = json.load(file)

            # Extract the speaker names from the JSON data
            speaker_names = set(word['speaker'] for segment in data['transcription'] for word in segment.get('words', []) if 'speaker' in word)

            # Iterate through segments and extract relevant information
            for segment in data['transcription']:
                # segment_id = segment['id']
                start_time = segment['timestamps']['from']
                end_time = segment['timestamps']['to']

                # Extract words spoken by each person
                texts = segment['text']
                words = texts.split()
                # Initialize the speaker name for this segment
                segment_speaker = None

                for word in words:
                    if segment['speaker']:
                        segment_speaker = segment['speaker']
                    if segment_speaker:
                        speaker_name = segment_speaker
                    else:
                        speaker_name = "unknown"

                    spoken_text = word if segment_speaker else f"{word} (unknown)"

                    all_table_data.append({
                        'Person': speaker_name,
                        'Sentence': spoken_text.strip(),
                        'Start Time': start_time,
                        'End Time': end_time,
                        'File Name': filename  # Store the file name
                    })

    # Merge consecutive sentences spoken by the same person in the final result
    merged_table_data = []
    current_speaker = None
    current_sentence = ""
    current_file_names = []

    for data in all_table_data:
        if current_speaker == data['Person']:
            current_sentence += " " + data['Sentence']
            current_file_names.append(data['File Name'])
        else:
            if current_sentence:
                merged_table_data.append({
                    'Person': current_speaker,
                    'Sentence': current_sentence,
                    'Start Time': current_start_time,
                    'End Time': current_end_time,
                    'File Names': ', '.join(current_file_names)  # Store multiple file names as a comma-separated string
                })
            current_speaker = data['Person']
            current_sentence = data['Sentence']
            current_start_time = data['Start Time']
            current_end_time = data['End Time']
            current_file_names = [data['File Name']]


    # Add the last merged sentence
    if current_sentence:
        merged_table_data.append({
            'Person': current_speaker,
            'Sentence': current_sentence,
            'Start Time': current_start_time,
            'End Time': current_end_time,
            'File Names': ', '.join(current_file_names)
        })

    # Create a DataFrame
    df = pd.DataFrame(merged_table_data)

    # Drop duplicate file names from the 'File Names' column
    df['File Names'] = df['File Names'].apply(lambda x: ', '.join(sorted(set(x.split(', ')))))

    # Calculate the number of words spoken by each person
    df['Word Count'] = df['Sentence'].apply(lambda x: len(x.split()))

    # Create a bag of words dictionary
    # List of words with possible wildcards
    bag_of_words = [
        "Community garden",
        "Food desert",
        "Food swamp",
        "food system",
        "insecurity",
        "health",
        "obese",
        "garden",
        "access",
        "urban",
        "poverty",
        "rural",
        "low income",
        "middle income",
        "prices",
        "minority",
        "Sovereignty",
        "Local",
        "affordable",
        "Vegetable",
        "Meat",
        "hung",
        "Nutrition",
        "Grow",
        "Gather",
        "Grocery",
        "Agriculture",
        "Climate change",
        "Usda",
        "Food",
        "Policy",
        "plant",
        "environment",
        "greenhouse gas",
        "organic"
    ]
    # Replace with your dictionary
    bag_of_words = [word.lower() for word in bag_of_words]

    # Create a dictionary with root words (without wildcards)
    root_word_dict = {get_lemma(word): '*' in word for word in bag_of_words}


    # Initialize a dictionary to store the first occurrence of words from the bag of words
    first_occurrence = {word: None for word in root_word_dict}

    # Iterate through the DataFrame to find the first occurrence of words
    prev = None
    for index, row in df.iterrows():
        words = row['Sentence'].split()
        for word in words:
            word = get_lemma(word)
            word = word.lower()
            if word in root_word_dict and first_occurrence[word] is None:
                first_occurrence[word] = row['Person']

            if prev and prev + word in root_word_dict and first_occurrence[prev + word] is None:
                first_occurrence[prev + word] = row['Person']

            prev = word
    ans = []
    word_counts_result = {speaker: 0 for speaker in preset_speakers}

    # Update the word counts for speakers who have spoken
    for person, group in df.groupby('Person'):
        word_count = group['Word Count'].sum()
        # Update the count for this person
        word_counts_result[person] = int(word_count)
    
    for word, person in first_occurrence.items():
        if person:
            ans.append(f"{person} spoke the word '{word}' first.")
            #print(f"{person} spoke the word '{word}' first.")
        else:
            ans.append(f"The word '{word}' was not spoken.")
            #print(f"The word '{word}' was not spoken.")

    # Prepare the separate results for word counts and first words spoken
    word_counts_result = {person: int(group['Word Count'].sum().item()) for person, group in df.groupby('Person')}
    first_words_spoken_result = first_occurrence

    id_str = get_id_str(DIR_NAME)
    cur_date_formatted = datetime.now().strftime('%Y-%m-%d')
    cur_time_formatted = datetime.now().strftime('%H:%M:%S')

    try:
        response = table.get_item(Key={'group_id': id_str})
        item = response.get('Item')

        if item:
            # Ensure current date exists directly within the item
            cur_date_data = item.setdefault(cur_date_formatted, {})
        else:
            # If the item does not exist, create a new structure for it
            cur_date_data = {}
            item = {
                'group_id': "Apple",
                cur_date_formatted: cur_date_data
            }

        # Update word counts and first words spoken in separate layers within the current date
        cur_date_data.setdefault('word_counts', {})
        cur_date_data['word_counts'][f"Results_{cur_time_formatted}"] = json.dumps(word_counts_result)
        
        cur_date_data.setdefault('first_words_spoken', {})
        if first_words_spoken_result:  # Check if the result is not empty before updating
            cur_date_data['first_words_spoken'][f"Results_{cur_time_formatted}"] = json.dumps(first_words_spoken_result)
        else:
            cur_date_data['first_words_spoken'][f"Results_{cur_time_formatted}"] = "{}"

        # Update or create the item in the table
        table.put_item(Item=item)

    except botocore.exceptions.ClientError as error:
        print(f"An error occurred: {error}")

    result = {
        'message': 'Analysis completed and stored in DynamoDB.',
        'word_counts': word_counts_result,
        'first_words_spoken': first_words_spoken_result
    }
    
    print(first_words_spoken_result)
    
    return jsonify(result)

@app.route('/check_server_working', methods=['GET'])
def check_server_working():
    res = {
        'message': 'Can connect and working'
    }
    return jsonify(res)

# @app.route('/word_concatenations', methods=['POST'])
# def word_concatenations():
#     parser = argparse.ArgumentParser(description="directory")
#     parser.add_argument("-d", "--directory", required=True, help="directory that will contain the dataset")
#     args = parser.parse_args()
#     DIR_NAME = args.directory

#     # Opening the appropriate JSON for the chunk of speech just recorded
#     filename = DIR_NAME + f'/recorded_data/chunk_{request.json['iteration']}.wav.json'
#     if os.path.exists(filename):
#         # Load the JSON data from the file
#         with open(filename, 'r') as file:
#             data = json.load(file)

#     # Creating a dictionary to store speakers' words
#     speakers_words = {}

#     # Extracting words spoken by each speaker
#     for word_info in data['segments'][0]['words']:
#         speaker = word_info['speaker']
#         word = word_info['text']
#         if speaker not in speakers_words:
#             speakers_words[speaker] = []
#         speakers_words[speaker].append(word)

#     # Converting lists of words into space-separated strings
#     for speaker in speakers_words:
#         speakers_words[speaker] = ' '.join(speakers_words[speaker])

#     # DynamoDB update logic (needed for saving to DB)
#     id_str = get_id_str(DIR_NAME)
#     cur_date_formatted = datetime.now().strftime('%Y-%m-%d')
#     cur_time_formatted = datetime.now().strftime('%H:%M:%S')

#     # Prepare the word concatenations
#     # Under "word_concatenations" in the DB are maps for each speaker, which then map to individual "results" containing the string of concatenated words
#     word_concats = {}
#     for speaker, words in speakers_words.items():
#         word_concats[speaker] = {
#             f"Results_{request.json['iteration']}": words
#         }

#     # The try-catch block saves the word concatenations to the DB
#     try:
#         # Fetch the current item based on group_id
#         response = table.get_item(Key={'group_id': id_str})
#         item = response.get('Item')

#         if item:
#             # Ensure current date's map exists directly within the item
#             if cur_date_formatted not in item:
#                 item[cur_date_formatted] = {}

#             # Add the new result to the 'word_concatenations' section within the current date
#             cur_date_data = item[cur_date_formatted]
#             cur_date_data.setdefault('word_concatenations', {})
            
#             # Merge new words with existing words for each speaker
#             # (we can't just set the 'word_concatenations' attribute equal to the word_concats variable otherwise we'll overwrite previous entries)
#             for speaker, new_results in word_concats.items():
#                 if speaker not in cur_date_data['word_concatenations']:
#                     cur_date_data['word_concatenations'][speaker] = new_results
#                 else:
#                     for key, new_words in new_results.items():
#                         cur_date_data['word_concatenations'][speaker][key] = new_words

#             # Update the item in the table
#             table.put_item(Item=item)
#         else:
#             # If the item does not exist, create a new one with the date directly under group_id
#             new_item = {
#                 'group_id': id_str,
#                 cur_date_formatted: {
#                     'word_concatenations': word_concats
#                 }
#             }
            
#             table.put_item(Item=new_item)

#     except botocore.exceptions.ClientError as error:
#         # Handle the exception
#         print(f"An error occurred: {error}")
    
#     result = {
#         'message': 'Word concatenations completed and stored in DynamoDB',
#         'word_concatenations': word_concatenations
#     }

#     return jsonify(result)

# @app.route('/emotion_check', methods=['POST'])
# def emotion_check():
#     parser = argparse.ArgumentParser(description="directory")
#     parser.add_argument("-d", "--directory", required=True, help="directory that will contain the dataset")
#     args = parser.parse_args()
#     DIR_NAME = args.directory

#     id_str = get_id_str(DIR_NAME)
#     response = table.get_item(Key={'group_id': id_str})
#     item = response.get('Item')

#     # Access the nested structure
#     cur_date_formatted = datetime.now().strftime('%Y-%m-%d')
#     word_concats = item.get(cur_date_formatted, {}).get('word_concatenations', {})
#     emotion_results = {}

#     for speaker, results in word_concats.items():
#         last_four = list(results.items())[-4:]
#         amount_of_recent = 0    # there needs to be 2 recent chunks to count it
#         all_words = ''
#         for iteration_num, words in last_four:
#             recently_transcribed_iteration = request.json['iteration'] / 15
#             which_iteration = iteration_num / 15
#             if which_iteration <= recently_transcribed_iteration and which_iteration > recently_transcribed_iteration - 4:
#                 all_words += words
#                 amount_of_recent += 1
    
#         # only counts if the last 2 out of 4 chunks include speech from this speaker
#         if amount_of_recent >= 2:
#             emotion_results[speaker] = {
#             f"Results_{request.json['iteration']}": all_words
#         }

#     # Finds the attribute for the current date and adds "emotion_results" if not there
#     cur_date_data = item[cur_date_formatted]
#     cur_date_data.setdefault('emotion_results', {})

#     # Adds the emotion data just collected to the speaker name attributes (or if they aren't there, adds a new speaker name entry)
#     for speaker, new_results in emotion_results.items():
#         if speaker not in cur_date_data['emotion_results']:
#             cur_date_data['emotion_results'][speaker] = new_results
#         else:
#             for key, emotion_data in new_results.items():
#                 cur_date_data['emotion_results'][speaker][key] = emotion_data
            
#     return jsonify({
#         'message': 'Emotion analysis completed and stored in DynamoDB',
#         'emotion_results': emotion_results
#     })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
