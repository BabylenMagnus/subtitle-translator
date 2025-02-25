from srt_translator import SubtitleTranslator
from srt_parser import parse_srt_file, save_srt_file
import os
import sys


def translate_srt_file(input_file: str, target_lang: str, source_lang: str = "English", output_file: str = None, test: bool = False) -> None:
    """
    Translate an SRT file from source language to target language.
    
    Args:
        input_file: Path to source SRT file
        target_lang: Target language for translation
        source_lang: Source language of the file (default: English)
        output_file: Path to save translated file (optional)
        test: If True, only translate first 200 lines (default: False)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    if output_file is None:
        output_file = input_file.replace(".srt", f".{target_lang.lower()}.srt")

    # Parse input SRT
    subtitles = parse_srt_file(input_file)

    # Initialize translator and translate
    translator = SubtitleTranslator(
        model="hf.co/RefalMachine/RuadaptQwen2.5-32B-Pro-Beta-GGUF:Q4_K_M",
        local=True,
        temperature=0.4
    )
    translated_subs = translator.translate_subtitles(
        subtitles=subtitles,
        target_lang=target_lang,
        source_lang=source_lang,
        test=test
    )

    # Save translated file
    save_srt_file(translated_subs, output_file)


def main():
    # Simple argument parsing
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    target_lang = sys.argv[2] if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    test = "--test" in sys.argv

    if input_file is None or target_lang is None:
        print("Usage: python translate_srt.py <input_file> <target_lang> [output_file] [--test]")
        sys.exit(1)

    translate_srt_file(input_file, target_lang, output_file=output_file, test=test)


if __name__ == "__main__":
    main()
