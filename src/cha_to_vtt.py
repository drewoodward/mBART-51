import re

def clean_cha_file(input_filepath, output_filepath):
    """
    Reads a .cha file and creates a new file with lines starting 
    with '%eng:' and '%aut:' removed.
    """
    try:
        # Using utf-8 encoding is generally safest for text files with special characters
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # We use lstrip() just in case there are accidental spaces before the '%'
                    cleaned_line = line.lstrip()
                    
                    if not (cleaned_line.startswith('%eng:') or cleaned_line.startswith('%aut:')):
                        outfile.write(line)
                        
        print(f"Success! Cleaned file saved to: {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def cha_to_vtt(cha_filepath, vtt_filepath):
    """
    Parses a .cha file, extracts timestamps and text, cleans CHAT metadata, 
    and outputs a Whisper fine-tuning ready .vtt file.
    """
    # Regex to catch the CHAT timestamp format (e.g., 470_2107)
    timestamp_pattern = re.compile(r'(\d+)_(\d+)')
    
    # Regex to identify the start of a speaker's line
    speaker_pattern = re.compile(r'^\*([A-Z]+):\s*(.*)')

    def format_time(ms_str):
        """Converts milliseconds to VTT timestamp format HH:MM:SS.mmm"""
        ms = int(ms_str)
        hrs = ms // 3600000
        mins = (ms % 3600000) // 60000
        secs = (ms % 60000) // 1000
        mils = ms % 1000
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{mils:03d}"

    def clean_chat_text(text):
        """Removes CHAT-specific transcription tags to leave clean training text."""
        # Remove comment blocks and metadata like [//], [///], [=! laughing]
        text = re.sub(r'\[.*?\]', '', text) 
        # Remove < and > used for overlapping or retractions
        text = re.sub(r'[<>]', '', text)     
        # Replace CHAT punctuation markers like +/. or +//. with standard periods
        text = re.sub(r'\+/?\.', '.', text)  
        # Remove hesitation markers like &e, &um, &a
        text = re.sub(r'&\w+', '', text)     
        # Remove pause markers (.) and (..)
        text = re.sub(r'\(\.\.?\)', '', text)
        # Remove trailing standalone language markers (s, s:eng) left by the transcriber
        text = re.sub(r'\b(s|s:eng|\+eng)\b', '', text)
        # Fix missing letters denoted by parentheses e.g. (th)em -> them
        text = re.sub(r'\((\w+)\)', r'\1', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)     
        
        return text.strip()

    try:
        with open(cha_filepath, 'r', encoding='utf-8') as infile, \
             open(vtt_filepath, 'w', encoding='utf-8') as outfile:
            
            outfile.write("WEBVTT\n\n")
            
            current_text = ""
            
            for line in infile:
                line = line.strip()
                
                # Skip CHAT file headers and metadata lines entirely
                if line.startswith('@') or line.startswith('%'):
                    continue
                
                # Check if it's a new speaker line
                speaker_match = speaker_pattern.match(line)
                if speaker_match:
                    # Append the text portion
                    current_text = speaker_match.group(2)
                else:
                    # It's a continuation line or a standalone timestamp
                    current_text += " " + line
                
                # Check if the current accumulated block contains a timestamp
                ts_match = timestamp_pattern.search(current_text)
                if ts_match:
                    start_ms, end_ms = ts_match.groups()
                    start_vtt = format_time(start_ms)
                    end_vtt = format_time(end_ms)
                    
                    # Strip the timestamp itself out of the text
                    clean_text_part = timestamp_pattern.sub('', current_text)
                    
                    # Run the CHAT cleaning functions
                    final_text = clean_chat_text(clean_text_part)
                    
                    # Only write if there's actual spoken text left
                    if final_text and final_text != ".":
                        outfile.write(f"{start_vtt} --> {end_vtt}\n")
                        outfile.write(f"{final_text}\n\n")
                    
                    # Reset the string block for the next speaker/timestamp
                    current_text = ""
                    
        print(f"Success! Cleaned VTT file saved to: {vtt_filepath}")

    except Exception as e:
        print(f"An error occurred: {e}")

# first we clean the CHAT file
input_file = 'miami/sastre1.cha'   
output_file = 'cleanCorpus/sastre1_cleaned.cha' 

clean_cha_file(input_file, output_file)

# next we convert to .vtt
input_cha = 'cleanCorpus/sastre1_cleaned.cha'
output_vtt = 'cleanCorpus/sastre1_perfect.vtt'

cha_to_vtt(input_cha, output_vtt)