#!/usr/bin/env python3
"""
Script Chunker for Storyboard Generation
Automatically splits YouTube scripts into scenes based on word count
Adds [SCENE XXX] markers every 10-12 words for precise image generation
"""

import re
import sys

try:
    import spacy
except ImportError:
    spacy = None

_SPACY_NLP = None

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

ACTION_SUBJECT_WORDS = {
    "he",
    "she",
    "they",
    "we",
    "i",
    "you",
    "chike",
    "amara",
    "adanna",
    "zainab",
}

ACTION_VERB_WORDS = {
    "asked",
    "buzzed",
    "called",
    "checked",
    "closed",
    "crumpled",
    "cut",
    "demanded",
    "entered",
    "excused",
    "grabbed",
    "handed",
    "interrupted",
    "laughed",
    "looked",
    "moved",
    "opened",
    "paused",
    "pointed",
    "pulled",
    "ran",
    "revealed",
    "said",
    "showed",
    "spun",
    "stepped",
    "stood",
    "turned",
    "walked",
    "whispered",
    "building",
    "living",
    "laughing",
}


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


def _get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if spacy is None:
        return None

    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _SPACY_NLP = nlp
    return _SPACY_NLP


def _split_sentences(text):
    nlp = _get_spacy_nlp()
    if nlp is None:
        # Fallback when spaCy is unavailable.
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


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


def _find_soft_break_within_range(words, start_idx, min_idx, target_idx, max_idx):
    """Find a natural break within a hard word budget."""
    if min_idx >= max_idx:
        return None

    connector_tokens = {
        "and",
        "but",
        "so",
        "because",
        "while",
        "when",
        "that",
        "which",
        "who",
        "as",
        "then",
    }

    # 1) Prefer sentence-ending punctuation near target.
    for idx in range(min(target_idx, max_idx - 1), min_idx - 1, -1):
        if words[idx].endswith((".", "!", "?", ";")):
            return idx + 1
    for idx in range(target_idx, max_idx):
        if words[idx].endswith((".", "!", "?", ";")):
            return idx + 1

    # 2) Then comma/colon/dash boundaries.
    for idx in range(min(target_idx, max_idx - 1), min_idx - 1, -1):
        if words[idx].endswith((",", ":", ";", "-", "—")):
            return idx + 1
    for idx in range(target_idx, max_idx):
        if words[idx].endswith((",", ":", ";", "-", "—")):
            return idx + 1

    # 3) Finally split before connectors.
    for idx in range(target_idx, max_idx):
        if _normalize_token(words[idx]) in connector_tokens and idx > start_idx + 2:
            return idx
    for idx in range(min(target_idx, max_idx - 1), min_idx, -1):
        if _normalize_token(words[idx]) in connector_tokens and idx > start_idx + 2:
            return idx

    return None


def _is_action_start(words, idx):
    if idx >= len(words):
        return False
    token = _normalize_token(words[idx])
    if token in ACTION_VERB_WORDS:
        return True
    if token in ACTION_SUBJECT_WORDS and idx + 1 < len(words):
        next_token = _normalize_token(words[idx + 1])
        return next_token in ACTION_VERB_WORDS
    return False


def _is_action_clause_start(words, idx, scene_start_idx):
    """Detect starts of a new action clause (subject + verb pattern)."""
    if idx >= len(words):
        return False

    token_raw = words[idx]
    token = _normalize_token(token_raw)

    if token in ACTION_SUBJECT_WORDS and idx + 1 < len(words):
        next_token = _normalize_token(words[idx + 1])
        return next_token in ACTION_VERB_WORDS

    # Name + action verb, e.g. "Chike grabbed"
    if token_raw and token_raw[0].isupper() and idx + 1 < len(words):
        next_token = _normalize_token(words[idx + 1])
        if next_token in ACTION_VERB_WORDS:
            return True

    # Allow imperative/verb-led opening at start of a scene.
    if idx == scene_start_idx and token in ACTION_VERB_WORDS:
        return True

    return False


def _find_second_action_clause_start(words, start_idx, end_idx):
    """Return index where the second action clause starts within the range."""
    action_clause_count = 0
    for idx in range(start_idx, end_idx):
        if _is_action_clause_start(words, idx, start_idx):
            action_clause_count += 1
            if action_clause_count >= 2:
                return idx
    return None


def _find_action_start(words, start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        if _is_action_start(words, idx):
            return idx
    return None


def _scene_has_action(scene_words):
    for idx in range(len(scene_words)):
        if _is_action_start(scene_words, idx):
            return True
    return False


def _merge_short_non_action_scenes(scene_word_chunks, min_words=6):
    if not scene_word_chunks:
        return scene_word_chunks

    merged = []
    i = 0
    while i < len(scene_word_chunks):
        current = scene_word_chunks[i]
        if len(current) < min_words and i + 1 < len(scene_word_chunks):
            nxt = scene_word_chunks[i + 1]
            # Preserve short action beats when consecutive action scenes are present.
            if not (_scene_has_action(current) and _scene_has_action(nxt)):
                current = current + nxt
                i += 1
        elif len(current) < min_words and merged:
            prev = merged[-1]
            # Merge tiny trailing/non-leading fragments back unless both are action beats.
            if not (_scene_has_action(prev) and _scene_has_action(current)):
                merged[-1] = prev + current
                i += 1
                continue
        merged.append(current)
        i += 1
    return merged


def _merge_non_action_leadins(scene_word_chunks, max_words):
    if not scene_word_chunks:
        return scene_word_chunks

    merged = []
    i = 0
    while i < len(scene_word_chunks):
        current = scene_word_chunks[i]
        if i + 1 < len(scene_word_chunks):
            nxt = scene_word_chunks[i + 1]
            current_last = current[-1] if current else ""
            if (
                current
                and not _scene_has_action(current)
                and _scene_has_action(nxt)
                and not current_last.endswith((".", "!", "?", ";"))
                and (len(current) + len(nxt)) <= max_words
            ):
                merged.append(current + nxt)
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged


def _rebalance_tiny_scene_starts(scene_word_chunks, max_words):
    if not scene_word_chunks:
        return scene_word_chunks

    rebalanced = [scene_word_chunks[0]]
    for current in scene_word_chunks[1:]:
        prev = rebalanced[-1]
        first_word = current[0] if current else ""
        prev_last = prev[-1] if prev else ""
        is_tiny = len(current) <= 3
        starts_lower = bool(first_word) and first_word[0].islower()
        prev_is_open = not prev_last.endswith((".", "!", "?", ";"))
        can_attach = (len(prev) + len(current)) <= (max_words + 3)

        # Avoid orphaned micro-scenes such as "real wife." / "years, Chike."
        if (is_tiny or (starts_lower and prev_is_open)) and can_attach and not _scene_has_action(current):
            rebalanced[-1] = prev + current
        else:
            rebalanced.append(current)
    return rebalanced


def _split_long_sentence_words(words, words_per_scene, min_words, max_words):
    chunks = []
    remaining = words[:]

    while remaining:
        if len(remaining) <= max_words:
            chunks.append(remaining)
            break

        split_at = _find_soft_break_within_range(
            words=remaining,
            start_idx=0,
            min_idx=min_words,
            target_idx=min(words_per_scene, len(remaining) - 1),
            max_idx=min(max_words, len(remaining)),
        )
        if split_at is None or split_at <= 0:
            split_at = max_words

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]

    return chunks


def split_into_scenes_contextual(
    text,
    words_per_scene=11,
    lookahead=4,
    min_words=None,
    max_words=None,
):
    """Split text with spaCy sentence boundaries + action-aware pacing."""
    if not text or not text.strip():
        return []

    if min_words is None:
        min_words = max(6, words_per_scene - 2)
    if max_words is None:
        max_words = max(words_per_scene + 1, min_words + 1)

    # 1) Sentence-first segmentation from spaCy.
    sentence_units = []
    for sentence in _split_sentences(text):
        words = sentence.split()
        if not words:
            continue
        if len(words) <= max_words:
            sentence_units.append(words)
            continue
        sentence_units.extend(
            _split_long_sentence_words(
                words=words,
                words_per_scene=words_per_scene,
                min_words=min_words,
                max_words=max_words,
            )
        )

    if not sentence_units:
        return []

    # 2) Build scenes with one-action-clause preference + tight pacing.
    scenes = []
    current_scene = []
    current_has_action = False

    for unit in sentence_units:
        unit_has_action = _scene_has_action(unit)
        unit_len = len(unit)
        current_len = len(current_scene)

        if not current_scene:
            current_scene = unit[:]
            current_has_action = unit_has_action
            continue

        should_split_for_action = current_has_action and unit_has_action and current_len >= min_words
        would_exceed_budget = (current_len + unit_len) > max_words

        if should_split_for_action or would_exceed_budget:
            scenes.append(current_scene)
            current_scene = unit[:]
            current_has_action = unit_has_action
        else:
            current_scene.extend(unit)
            current_has_action = current_has_action or unit_has_action

    if current_scene:
        scenes.append(current_scene)

    scenes = _rebalance_tiny_scene_starts(scenes, max_words=max_words)

    formatted_scenes = []
    scene_number = 1
    for scene_words in scenes:
        scene_text = " ".join(scene_words)
        formatted_scenes.append(f"[SCENE {scene_number:03d}] {scene_text}")
        scene_number += 1

    return formatted_scenes


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
