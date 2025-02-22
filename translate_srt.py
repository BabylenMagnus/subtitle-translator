from srt_translator import SubtitleTranslator
from srt_parser import parse_srt_file, save_srt_file
import os
import sys
import traceback


def translate_file(input_file: str, output_file: str = None) -> None:
    """
    Translate an SRT file from English to Russian.
    
    Args:
        input_file: Path to source SRT file
        output_file: Path to save translated file (optional)
    """
    print(f"\nStarting translation process...")
    print(f"Input file: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    if output_file is None:
        output_file = input_file.replace(".srt", ".translated.srt")
    print(f"Output will be saved to: {output_file}")

    # Initialize translator with specified model and parameters
    print("\nInitializing translator...")
    translator = SubtitleTranslator(
        model="deepseek-r1-distill-llama-70b",
        local=False,
        temperature=0.4
    )
    print("Translator initialized successfully")

    # Parse input SRT
    print(f"\nParsing {input_file}...")
    subtitles = parse_srt_file(input_file)
    print(f"Found {len(subtitles)} subtitles")

    # Translate
    print("\nTranslating...")
    translated_subs = translator.translate_subtitles(
        subtitles=subtitles,
        source_lang="English",
        target_lang="Russian"
    )

    # Save translated file
    print(f"\nSaving to {output_file}...")
    save_srt_file(translated_subs, output_file)
    print("Translation completed!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python translate_srt.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        translate_file(input_file, output_file)
    except Exception as e:
        print(f"\nError occurred:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
