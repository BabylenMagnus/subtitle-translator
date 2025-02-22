from srt_translator import SubtitleTranslator
import re
from typing import List, Dict


def parse_srt_file(file_path: str) -> List[Dict]:
    """
    Parse an SRT file into a list of subtitle dictionaries.
    
    Args:
        file_path: Path to the SRT file
        
    Returns:
        List of dictionaries with index, timestamp, and text
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into subtitle blocks
    subtitle_blocks = re.split(r'\n\n+', content.strip())
    subtitles = []

    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
            timestamp = lines[1]
            text = '\n'.join(lines[2:])

            subtitles.append({
                'index': index,
                'timestamp': timestamp,
                'text': text
            })
        except (ValueError, IndexError):
            continue

    return subtitles


def save_srt_file(subtitles: List[Dict], output_path: str) -> None:
    """
    Save subtitles to an SRT file.
    
    Args:
        subtitles: List of subtitle dictionaries with index, timestamp, text
        output_path: Path where to save the SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            # Write index
            f.write(f"{subtitle['index']}\n")
            
            # Write timestamp
            f.write(f"{subtitle['timestamp']}\n")
            
            # Write text
            f.write(f"{subtitle['text']}\n")
            
            # Write blank line between subtitles
            f.write('\n')


def get_plain_text(file_path):
    """
    Extract just the text content from an SRT file, 
    skipping timestamps and indices.
    """
    subtitles = parse_srt_file(file_path)
    return '\n'.join(sub['text'] for sub in subtitles)


def translate_srt(file_path: str, target_lang: str, output_path: str):
    """
    Translate an SRT file to target language and save to new file.
    
    Args:
        file_path: Path to source SRT file
        target_lang: Target language for translation
        output_path: Path to save translated SRT file
    """
    # Parse original subtitles
    subtitles = parse_srt_file(file_path)

    # Initialize translator
    translator = SubtitleTranslator(
        temperature=0.3, model="deepseek-r1:32b", local=True, context_lines=10
    )

    # Translate subtitles
    translated_subs = translator.translate_subtitles(
        subtitles=subtitles, target_lang=target_lang
    )

    # Write translated SRT file
    save_srt_file(translated_subs, output_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        print("Usage: python srt_parser.py <input_srt> <target_lang> <output_srt>")
        sys.exit(1)

    input_file = sys.argv[1]
    target_lang = sys.argv[2]
    output_file = sys.argv[3]

    try:
        translate_srt(input_file, target_lang, output_file)
        print(f"Successfully translated to {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
