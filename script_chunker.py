#!/usr/bin/env python3
"""
Script Chunker for Storyboard Generation
Automatically splits YouTube scripts into scenes based on word count
Adds [SCENE XXX] markers every 10-12 words for precise image generation
"""

import re
import sys


VISUAL_CUE_WORDS = {
    "appears",
    "arrives",
    "camera",
    "closes",
    "cut",
    "enters",
    "exits",
    "focuses",
    "looks",
    "monitor",
    "moves",
    "opens",
    "pans",
    "points",
    "reveals",
    "runs",
    "shows",
    "stands",
    "starts",
    "turns",
    "walks",
    "watches",
    "zooms",
}

VISUAL_CUE_PHRASES = (
    ("cuts", "to"),
    ("cut", "to"),
    ("camera", "pans"),
    ("camera", "zooms"),
    ("zoom", "in"),
    ("zoom", "out"),
)


def clean_text(text):
    """Remove script formatting markers and clean up text"""
    # Remove time codes like [0:00-0:30]
    text = re.sub(r'\[[\d:]+\-[\d:]+\]', '', text)
    # Remove section markers like [HOOK 1], [OPENING HOOK], etc.
    text = re.sub(r'\[(?:HOOK|OPENING|CLOSING|MAIN STORY|THE END|FADE OUT).*?\]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_scenes(text, words_per_scene=11):
    """
    Split text into scenes based on word count
    
    Args:
        text: The script text to split
        words_per_scene: Target words per scene (default: 11 for 5 seconds of narration)
    
    Returns:
        List of scene chunks with scene numbers
    """
    # Split into words
    words = text.split()
    
    scenes = []
    current_scene = []
    scene_number = 1
    
    for word in words:
        current_scene.append(word)
        
        # When we hit target word count, create a scene
        if len(current_scene) >= words_per_scene:
            # Join the words
            scene_text = ' '.join(current_scene)
            
            # Add scene marker
            scenes.append(f"[SCENE {scene_number:03d}] {scene_text}")
            
            # Reset for next scene
            current_scene = []
            scene_number += 1
    
    # Handle remaining words
    if current_scene:
        scene_text = ' '.join(current_scene)
        scenes.append(f"[SCENE {scene_number:03d}] {scene_text}")
    
    return scenes


def _normalize_token(word):
    return re.sub(r"^[^\w]+|[^\w]+$", "", word).lower()


def _find_cue_index(words, start_idx, end_idx):
    normalized = [_normalize_token(word) for word in words]

    for idx in range(start_idx, end_idx):
        token = normalized[idx]
        if token in VISUAL_CUE_WORDS:
            return idx

        for phrase in VISUAL_CUE_PHRASES:
            phrase_len = len(phrase)
            if idx + phrase_len > len(normalized):
                continue
            if tuple(normalized[idx : idx + phrase_len]) == phrase:
                return idx
    return None


def _find_punctuation_break(words, start_idx, end_idx):
    for idx in range(end_idx - 1, start_idx - 1, -1):
        if words[idx].endswith((".", "!", "?", ";")):
            return idx + 1
    return None


def split_into_scenes_contextual(
    text,
    words_per_scene=11,
    lookahead=4,
    min_words=None,
    max_words=None,
):
    """
    Split text using a hybrid strategy:
    1) target a word count
    2) prefer action/visual cue boundaries near the target
    3) fallback to punctuation
    4) hard split at max_words
    """
    words = text.split()
    if not words:
        return []

    if min_words is None:
        min_words = max(6, words_per_scene - 3)
    if max_words is None:
        max_words = max(words_per_scene + 6, min_words + 2)

    scenes = []
    scene_number = 1
    idx = 0
    total = len(words)

    while idx < total:
        min_split = min(total, idx + min_words)
        target_split = min(total, idx + words_per_scene)
        hard_split = min(total, idx + max_words)

        if target_split >= total:
            split_at = total
        else:
            cue_window_end = min(total, target_split + lookahead + 1)
            cue_idx = _find_cue_index(words, target_split, cue_window_end)

            if cue_idx is not None and cue_idx >= min_split:
                split_at = cue_idx
            else:
                punct_split = _find_punctuation_break(words, min_split, hard_split)
                split_at = punct_split if punct_split is not None else hard_split

        if split_at <= idx:
            split_at = min(total, idx + words_per_scene)
            if split_at <= idx:
                split_at = total

        scene_text = " ".join(words[idx:split_at])
        scenes.append(f"[SCENE {scene_number:03d}] {scene_text}")
        scene_number += 1
        idx = split_at

    return scenes


def process_script(input_file, output_file=None, words_per_scene=11, mode="basic"):
    """
    Process a script file and output scene-marked version
    
    Args:
        input_file: Path to input script file
        output_file: Path to output file (optional)
        words_per_scene: Words per scene (default: 11)
        mode: Splitting mode: "basic" or "context-aware" (default: basic)
    """
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Count original words
    total_words = len(cleaned_text.split())
    
    # Split into scenes
    if mode == "context-aware":
        scenes = split_into_scenes_contextual(cleaned_text, words_per_scene)
    else:
        scenes = split_into_scenes(cleaned_text, words_per_scene)
    
    # Calculate statistics
    total_scenes = len(scenes)
    estimated_runtime = (total_scenes * 5) / 60  # 5 seconds per scene
    
    # Create output
    output_text = '\n'.join(scenes)
    
    # Determine output destination
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"✅ Processed script saved to: {output_file}")
    else:
        print(output_text)
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 SCRIPT ANALYSIS")
    print("="*60)
    print(f"Total words: {total_words}")
    print(f"Total scenes: {total_scenes}")
    print(f"Mode: {mode}")
    print(f"Words per scene: ~{total_words/total_scenes:.1f}")
    print(f"Estimated runtime: {estimated_runtime:.1f} minutes")
    print("="*60)
    
    return scenes

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(
            "Usage: python script_chunker.py <input_file> "
            "[output_file] [words_per_scene] [mode]"
        )
        print("\nExamples:")
        print("  python script_chunker.py script.txt")
        print("  python script_chunker.py script.txt chunked_script.txt")
        print("  python script_chunker.py script.txt chunked_script.txt 12")
        print("  python script_chunker.py script.txt chunked_script.txt 12 context-aware")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    words_per_scene = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    mode = sys.argv[4] if len(sys.argv) > 4 else "basic"

    if mode not in {"basic", "context-aware"}:
        print("❌ Error: mode must be 'basic' or 'context-aware'")
        sys.exit(1)
    
    process_script(input_file, output_file, words_per_scene, mode)

if __name__ == "__main__":
    main()
