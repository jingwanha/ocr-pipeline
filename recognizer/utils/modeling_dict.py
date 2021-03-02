import re

def create_dict(text_files):
    character_set = set()
    batch_max_length = 0

    for text_file in text_files:
        with open(text_file, 'r') as f:
            texts = f.readlines()
            for text in texts:
                text = re.sub('[\n\s]', '', text)
                batch_max_length = max([batch_max_length, len(text)])
                character_set.update([c for c in text])

    characters = list(character_set)
    characters.sort()

    return characters